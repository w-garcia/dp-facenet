from hashlib import sha256
from custom_logging import get_logger

import numpy as np

np_dtype = np.dtype('float32')

g_logger = get_logger(__name__)


def query_hop_skip_jump(A_GT, oracle, **kwargs):
    """
    Public interface for query Hop Skip Jump algorithm. Filters away hyper params
    :param A_GT: Adversary's known registered data.
    :param oracle: THE ORACLE with method 'query' implemented
    :param kwargs: dict of data for logistical purposes but vary between domains (credentials etc)
    :return: D_A
    """
    hsja = QueryHopSkipJumpAttack(**kwargs)
    return hsja.generate_dataset(A_GT, oracle)


def clip_image(image, clip_min, clip_max):
    """ Clip an image, or an image batch, with upper and lower threshold. """
    return np.minimum(np.maximum(clip_min, image), clip_max)


def compute_distance(x_ori, x_pert, constraint='l2'):
    """ Compute the distance between two images. """
    if constraint == 'l2':
        dist = np.linalg.norm(x_ori - x_pert)
    elif constraint == 'linf':
        dist = np.max(abs(x_ori - x_pert))
    return dist


def approximate_gradient(decision_function, sample, num_evals,
                         delta, constraint, shape, clip_min, clip_max):
    """ Gradient direction estimation """
    # Generate random vectors.
    noise_shape = [num_evals] + list(shape)
    if constraint == 'l2':
        rv = np.random.randn(*noise_shape)
    elif constraint == 'linf':
        rv = np.random.uniform(low=-1, high=1, size=noise_shape)

    axis = tuple(range(1, 1 + len(shape)))
    rv = rv / np.sqrt(np.sum(rv ** 2, axis=axis, keepdims=True))
    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, clip_min, clip_max)
    rv = (perturbed - sample) / delta

    # query the model.
    decisions = decision_function(perturbed)
    decision_shape = [len(decisions)] + [1] * len(shape)
    fval = 2 * decisions.astype(np_dtype).reshape(decision_shape) - 1.0

    # Baseline subtraction (when fval differs)
    if np.mean(fval) == 1.0:  # label changes.
        gradf = np.mean(rv, axis=0)
    elif np.mean(fval) == -1.0:  # label not change.
        gradf = - np.mean(rv, axis=0)
    else:
        fval = fval - np.mean(fval)
        gradf = np.mean(fval * rv, axis=0)

    # Get the gradient direction.
    gradf = gradf / np.linalg.norm(gradf)

    return gradf


def project(original_image, perturbed_images, alphas, shape, constraint):
    """ Projection onto given l2 / linf balls in a batch. """
    alphas_shape = [len(alphas)] + [1] * len(shape)
    alphas = alphas.reshape(alphas_shape)
    if constraint == 'l2':
        projected = (1 - alphas) * original_image + alphas * perturbed_images
    elif constraint == 'linf':
        projected = clip_image(
            perturbed_images,
            original_image - alphas,
            original_image + alphas
        )
    return projected


def binary_search_batch(original_image, perturbed_images, decision_function,
                        shape, constraint, theta):
    """ Binary search to approach the boundary. """

    # Compute distance between each of perturbed image and original image.
    dists_post_update = np.array([
        compute_distance(
            original_image,
            perturbed_image,
            constraint
        )
        for perturbed_image in perturbed_images])

    # Choose upper thresholds in binary searchs based on constraint.
    if constraint == 'linf':
        highs = dists_post_update
        # Stopping criteria.
        thresholds = np.minimum(dists_post_update * theta, theta)
    else:
        highs = np.ones(len(perturbed_images))
        thresholds = theta

    lows = np.zeros(len(perturbed_images))

    while np.max((highs - lows) / thresholds) > 1:
        # projection to mids.
        mids = (highs + lows) / 2.0
        mid_images = project(original_image, perturbed_images,
                             mids, shape, constraint)

        # Update highs and lows based on model decisions.
        decisions = decision_function(mid_images)
        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)

    out_images = project(original_image, perturbed_images,
                         highs, shape, constraint)

    # Compute distance of the output image to select the best choice.
    # (only used when stepsize_search is grid_search.)
    dists = np.array([
        compute_distance(
            original_image,
            out_image,
            constraint
        )
        for out_image in out_images])
    idx = np.argmin(dists)

    dist = dists_post_update[idx]
    out_image = out_images[idx]
    return out_image, dist


def geometric_progression_for_stepsize(x, update, dist, decision_function,
                                       current_iteration):
    """ Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary.
    """
    epsilon = dist / np.sqrt(current_iteration)
    step = 0
    while True:
        if step > 1000:
            g_logger.warning("Hit cutoff for progression steps, quitting...")
            return None

        updated = x + epsilon * update
        success = decision_function(updated[None])[0]
        if success:
            break
        else:
            epsilon = epsilon / 2.0
        step += 1

    return epsilon


def select_delta(dist_post_update, current_iteration,
                 clip_max, clip_min, d, theta, constraint):
    """
    Choose the delta at the scale of distance
     between x and perturbed sample.
    """
    if current_iteration == 1:
        delta = 0.1 * (clip_max - clip_min)
    else:
        if constraint == 'l2':
            delta = np.sqrt(d) * theta * dist_post_update
        elif constraint == 'linf':
            delta = d * theta * dist_post_update

    return delta


class QueryHopSkipJumpAttack:
    def __init__(self, **kwargs):
        self.structural_kwargs = [
            'stepsize_search',
            'clip_min',
            'clip_max',
            'constraint',
            'num_iterations',
            'initial_num_evals',
            'max_num_evals',
            'batch_size',
            'verbose',
            'gamma',
        ]
        self.logger = get_logger(__name__)
        self.queries = 0

    def parse_params(self,
                     y_target=None,
                     sample_target=None,
                     initial_num_evals=5,
                     max_num_evals=40,
                     stepsize_search='geometric_progression',
                     num_iterations=10,
                     gamma=1.0,
                     constraint='l2',
                     batch_size=16,
                     verbose=True,
                     clip_min=0.,
                     clip_max=1.,
                     max_return_size=1,
                     queries_cutoff=1000):
        """
        :param y: A tensor of shape (1, nb_classes) for true labels.
        :param y_target:  A tensor of shape (1, nb_classes) for target labels.
        Required for targeted attack.
        :param sample_target: A tensor of shape (1, **image shape) for initial
        target images. Required for targeted attack.
        :param initial_num_evals: initial number of evaluations for
                                  gradient estimation.
        :param max_num_evals: maximum number of evaluations for gradient estimation.
        :param stepsize_search: How to search for stepsize; choices are
                                'geometric_progression', 'grid_search'.
                                'geometric progression' initializes the stepsize
                                 by ||x_t - x||_p / sqrt(iteration), and keep
                                 decreasing by half until reaching the target
                                 side of the boundary. 'grid_search' chooses the
                                 optimal epsilon over a grid, in the scale of
                                 ||x_t - x||_p.
        :param num_iterations: The number of iterations.
        :param gamma: The binary search threshold theta is gamma / d^{3/2} for
                       l2 attack and gamma / d^2 for linf attack.
        :param constraint: The distance to optimize; choices are 'l2', 'linf'.
        :param batch_size: batch_size for model prediction.
        :param verbose: (boolean) Whether distance at each step is printed.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # ignore the y and y_target argument
        self.y_target = y_target
        self.sample_target = sample_target
        self.initial_num_evals = initial_num_evals
        self.max_num_evals = max_num_evals
        self.stepsize_search = stepsize_search
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.constraint = constraint
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.verbose = verbose
        self.max_return_size = max_return_size
        self.queries_cutoff = queries_cutoff

    def generate_dataset(self, A_GT, oracle, **kwargs):
        """
        Generate adversarial images in a for loop.
        :param y: An array of shape (n, nb_classes) for true labels.
        :param y_target:  An array of shape (n, nb_classes) for target labels.
        Required for targeted attack.
        :param image_target: An array of shape (n, **image shape) for initial
        target images. Required for targeted attack.

        See parse_params for other kwargs.

        """

        self.parse_params(**kwargs)

        x_adv = []
        text_to_f_dict = {False: 0, True: 1}

        if 'sample_target' in kwargs and kwargs['sample_target'] is not None:
            sample_target = np.copy(kwargs['sample_target'])
        else:
            sample_target = None
        if 'y_target' in kwargs and kwargs['y_target'] is not None:
            y_target = np.copy(kwargs['y_target'])
        else:
            y_target = None

        A_GT = list(A_GT)
        i = 0
        while len(x_adv) != self.max_return_size:
            if len(A_GT) == 0 or self.queries >= self.queries_cutoff:
                break

            x_single = A_GT.pop()
            dec_i = oracle.query(x_single)
            if dec_i == False:
                # Need a correctly classified sample to start.
                continue

            if sample_target is not None:
                single_img_target = np.expand_dims(sample_target[i], axis=0)
                kwargs['image_target'] = single_img_target
            if y_target is not None:
                single_y_target = np.expand_dims(y_target[i], axis=0)
                kwargs['y_target'] = single_y_target

            adv_sample = self.generate(x_single, oracle)
            if adv_sample is None:
                continue

            x_adv.append(adv_sample)

            i += 1

        self.logger.warning(f"Left D_A creaation with {self.queries} queries.")
        return x_adv, self.queries

    def generate(self, x, oracle, **kwargs):
        """
            :param x: A tensor with the inputs.
            :param kwargs: See `parse_params`
            """
        shape = [int(i) for i in x.shape]

        # Set shape and d.
        self.shape = shape
        self.d = int(np.prod(shape))

        # Set binary search threshold.
        if self.constraint == 'l2':
            self.theta = self.gamma / (np.sqrt(self.d) * self.d)
        else:
            self.theta = self.gamma / (self.d * self.d)

        return self._hsja(oracle, x, True, **kwargs)

    def initialize(self, decision_function, sample):
        """
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0

        # Find a misclassified random noise.
        while True:
            random_noise = np.random.uniform(self.clip_min, self.clip_max, size=self.shape)
            success = decision_function(random_noise[None])[0]
            if success:
                break
            num_evals += 1
            if num_evals > 400:
                message = "Initialization failed! Try to use a misclassified image as `target_image`"
                self.logger.error(message)
                return None

        # Binary search to minimize l2 distance to original image.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
            mid = (high + low) / 2.0
            blended = (1 - mid) * sample + mid * random_noise
            success = decision_function(blended[None])[0]
            if success:
                high = mid
            else:
                low = mid

        self.logger.info(f"Initialization took {self.queries} queries")
        initialization = (1 - high) * sample + high * random_noise
        return initialization

    def _hsja(self, oracle, sample_i, original_label, target_label=None, target_sample=None, **kwargs):
        """
        Main algorithm for HopSkipJumpAttack.

        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param sample_i: input image. Without the batchsize dimension.
        :param target_label: integer for targeted attack,
          None for nontargeted attack. Without the batchsize dimension.
        :param target_image: an array with the same size as sample, or None.
          Without the batchsize dimension.


        Output:
        perturbed image.

        """

        def decision_function(samples):
            """
            Decision function output 1 on the desired side of the boundary,
            0 otherwise.
            """
            samples = clip_image(samples, self.clip_min, self.clip_max)
            
            dec = []
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i: i+self.batch_size]
                dec_i = oracle.query(batch)
                dec.append(dec_i)

            dec = np.concatenate(dec, axis=0)
            self.queries += len(samples)

            if target_label is None:
                return (np.asarray(dec) != original_label).astype(int)
            else:
                return (np.asarray(dec) == target_label).astype(int)

        # Initialize.
        if target_sample is None:
            perturbed = self.initialize(decision_function, sample_i)
            if perturbed is None:
                return None
        else:
            perturbed = target_sample

        # Project the initialization to the boundary.
        perturbed, dist_post_update = binary_search_batch(sample_i,
                                                          np.expand_dims(perturbed, 0),
                                                          decision_function,
                                                          self.shape,
                                                          self.constraint,
                                                          self.theta)

        dist = compute_distance(perturbed, sample_i, self.constraint)

        for j in np.arange(self.num_iterations):
            current_iteration = j + 1

            # Choose delta.
            delta = select_delta(dist_post_update, current_iteration,
                                 self.clip_max, self.clip_min, self.d,
                                 self.theta, self.constraint)

            # Choose number of evaluations.
            num_evals = int(min([self.initial_num_evals * np.sqrt(j + 1),
                                 self.max_num_evals]))
            self.logger.debug(f"Selected {num_evals} evaluations at step {j}.")

            # approximate gradient.
            gradf = approximate_gradient(decision_function, perturbed, num_evals,
                                         delta, self.constraint, self.shape,
                                         self.clip_min, self.clip_max)
            if self.constraint == 'linf':
                update = np.sign(gradf)
            else:
                update = gradf

            # search step size.
            if self.stepsize_search == 'geometric_progression':
                # find step size.
                epsilon = geometric_progression_for_stepsize(perturbed,
                                                             update, dist, decision_function, current_iteration)

                self.logger.debug("Passed geometric progression")
                if epsilon is None:
                    return None

                # Update the sample.
                perturbed = clip_image(perturbed + epsilon * update,
                                       self.clip_min, self.clip_max)

                # Binary search to return to the boundary.
                perturbed, dist_post_update = binary_search_batch(sample_i,
                                                                  perturbed[None],
                                                                  decision_function,
                                                                  self.shape,
                                                                  self.constraint,
                                                                  self.theta)
                self.logger.debug("Passed binary search")

            elif self.stepsize_search == 'grid_search':
                # Grid search for stepsize.
                epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
                epsilons_shape = [20] + len(self.shape) * [1]
                perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                perturbeds = clip_image(perturbeds, self.clip_min, self.clip_max)
                idx_perturbed = decision_function(perturbeds)

                if np.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                    perturbed, dist_post_update = binary_search_batch(sample_i,
                                                                      perturbeds[idx_perturbed],
                                                                      decision_function,
                                                                      self.shape,
                                                                      self.constraint,
                                                                      self.theta)

            # compute new distance.
            dist = compute_distance(perturbed, sample_i, self.constraint)
            if self.verbose:
                self.logger.debug('iteration: {:d}, {:s} distance {:.4E}'.format(j + 1, self.constraint, dist))

        # perturbed = np.expand_dims(perturbed, 0)
        return perturbed

