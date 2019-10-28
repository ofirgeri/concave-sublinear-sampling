"""
Implementation of a sampling sketch for concave sublinear functions for
frequencies from the NeurIPS 2019 paper:
Sampling Sketches for Concave Sublinear Functions of Frequencies
by Edith Cohen and Ofir Geri

The full version of the paper is available at:
https://arxiv.org/abs/1907.02218
or in the supplementary material in the NeurIPS 2019 proceedings.

This is a proof-of-concept implementation, which has not been optimized for
performance.

The main sketch from the paper is implemented as the class MainSketch below.
The function estimate uses this sketch to produce a sample from a list of data
elements, and shows how to estimate the f-statistics of the data using the
sample (including the second pass needed to compute the inverse-probability
estimators).
"""

import hashlib
import math
import numpy
import numpy.random
import scipy.integrate as integrate
import scipy.special

# The hash functions used for sampling
# Some (but not all) of the randomness used for sampling comes from the output
# of these hash functions, so in order to get different samples across runs,
# a different seed should be used each time (in the function estimate below).

def hashed_float(s):
    """
    Returns a float in the range uniformly in (0, 1] based on the SHA-256 hash
    of the string.
    """
    number = int(hashlib.sha256(s).hexdigest(), 16)
    return (number + 1) / float(2 ** 256)

def exp_hashed(s):
    """
    A hash function mapping strings into a float drawn from Exp(1).
    """
    return -1.0 * math.log(hashed_float(s))

# Sketch structure implementaion

class BottomK:
    """
    The bottom-k structure (Algorithm 1 in the full version).
    """
    
    def __init__(self, k):
        """
        Initializes an empty sketch of a given size.

        Parameters:
        k: The size of the sketch (number of elements stored)
        """
        # The sketch size
        self.k = k
        # A dictionary containing the data elements stored in the sketch.
        # For elements of the form (key, value), the dictionary maps a key
        # to its minimum value in the input elements.
        self.elements = {}

    def process(self, key, value):
        """
        Processes a data element (key, value) into the sketch.

        Parameters:
        key: The key of the element
        value: The value of the element
        """
        if key in self.elements.keys():
            self.elements[key] = min(self.elements[key], value)
        else:
            self.elements[key] = value

        # If the sketch size is greater than k, remove the largest elements
        # TODO (possible optimization): remove the largest elements only when
        # the sketch reaches the size 2k instead of k + 1.
        if len(self.elements.keys()) > self.k:
            sorted_items = sorted(self.elements.items(), key=lambda x: x[1])
            for i in xrange(self.k, len(sorted_items)):
                del self.elements[sorted_items[i][0]]
            return sorted_items[self.k]
        
        return None

    def merge(self, another):
        """
        Merges another bottom-k sketch (of the same size) into this sketch.

        Parameters:
        another: The other bottom-k sketch (all elements of another will be
            added into this sketch, but another will not be changed).
        """
        if another.k != self.k:
            raise Exception("merging bottom-k of different size")
        for key, value in another.elements.items():
            BottomK.process(self, key, value)

class Ppswor(BottomK):
    """
    A PPSWOR sampling sketch (Algorithm 2 in the full version).
    """
    
    def process(self, key, value):
        """
        Processes a data element (key, value) into the sketch.

        Parameters:
        key: The key of the element
        value: The value of the element
        """
        score = numpy.random.exponential(1.0 / value)
        BottomK.process(self, key, score)

class SumMax(BottomK):
    """
    The SumMax sampling sketch (Algorithm 3 in the full version).
    """

    # The string used to separate the primary and secondary key when creating
    # a string to pass as input to the hash function
    KEY_SEP = "!@#$"

    def __init__(self, k, hash_seed):
        """
        Initializes an empty sketch of a given size.

        Parameters:
        k: The size of the sketch (number of elements stored)
        hash_seed: A string used to seed the hash function. This determines all
            the randomness used for sampling.
        """
        # The sketch size
        self.k = k
        # The hash seed
        self.hash_seed = str(hash_seed)
        # A dictionary containing the data elements stored in the sketch.
        # For elements of the form (key, value), the dictionary maps a key
        # to its minimum value in the input elements.
        self.elements = {}
    
    def process(self, key, value):
        """
        Processes a data element (key, value) into the sketch.

        Parameters:
        key: The key of the element -- a tuple (primary key, secondary key).
            The primary key should not be the empty string or include the string
            "!@#$" to avoid hash collisions.
        value: The value of the element
        """
        if str(key[0]) == "" or SumMax.KEY_SEP in str(key[0]):
            raise Exception("Primary key is empty or contains forbidden string")
        score = exp_hashed(self.hash_seed + str(key[0]) + SumMax.KEY_SEP
                            + str(key[1])) / value
        BottomK.process(self, key[0], score)

    def merge(self, another):
        """
        Merges another SumMax sketch (of the same size) into this sketch. Both
        sketches need to use the same hash function (i.e., the same hash seed).

        Parameters:
        another: The other SumMax sketch (all elements of another will be
            added into this sketch, but another will not be changed).
        """
        if another.k != self.k:
            raise Exception("merging SumMax of different size")
        if another.hash_seed != self.hash_seed:
            raise Exception("merging SumMax with different hash functions")
        for key, value in another.elements.items():
            BottomK.process(self, key, value)

class MainSketch:
    """
    The main sampling sketch (Algorithms 4 and 5 in the full version).
    """
    
    def __init__(self, k, eps, funcA, funcB, hash_seed):
        """
        Initializes an empty sketch of a given size.

        Parameters:
        k: The size of the output sample
        eps: The parameter epsilon (trades off element processing time and
            variance)
        funcA: The function A for the desired soft concave sublinear function.
            Depends on the inverse complement Laplace transform of the function.
        funcB: The function B for the desired soft concave sublinear function.
            Depends on the inverse complement Laplace transform of the function.
        hash_seed: A string used to seed the hash function. This determines
            some of the randomness used for sampling.
        """
        
        self.k = k
        self.eps = float(eps)
        self.funcA = funcA
        self.funcB = funcB

        # The SumMax sample
        self.summax = SumMax(k, hash_seed)
        # The Sideline structure (elements not processed yet by the SumMax)
        self.sideline = {}
        # The sum of values of all elements
        self.sum = 0
        # The PPSWOR sample
        self.ppswor = Ppswor(k)
        # The threshold gamma (between the PPSWOR and SumMax samples)
        self.gamma = float("inf")
        # The number of output elements per input element
        self.r = int(math.ceil(self.k / self.eps))

        # Statistics on the size of the sketch
        # The maximum number of elements ever stored in the Sideline structure
        self.max_sideline_size = 0
        # The maximum number of distinct input keys ever stored in Sideline
        self.max_distinct_keys_sideline = 0
        # The maximum number of elements ever stored anywhere in the sketch
        # (in either the SumMax, PPSWOR, or Sideline structures)
        self.max_total_size = 0
        # The maxmimum number of distinct input keys ever stored by the sketch
        self.max_distinct_total_size = 0

    def process(self, key, value):
        """
        Processes a data element (key, value) into the sketch. See Algorithm 4
        (in the full version) for details.

        Parameters:
        key: The key of the element. Should not be the empty string or include
            the string "!@#$" to avoid hash collisions.
        value: The value of the element
        """
        self.ppswor.process(key, value)
        self.sum += value
        self.gamma = (2.0 * self.eps) / self.sum
        for i in xrange(self.r):
            yi = numpy.random.exponential(1.0 / value)
            if (key, i) in self.sideline.keys():
                self.sideline[(key, i)] = min(yi, self.sideline[(key, i)])
            else:
                self.sideline[(key, i)] = yi
        for sideline_key, sideline_val in self.sideline.items():
            if sideline_val >= self.gamma:
                del self.sideline[sideline_key]
                integral_res = self.funcA(sideline_val)
                if integral_res > 0:
                    self.summax.process(sideline_key, integral_res)

        # Optimization: remove unneeded elements in Sideline
        if len(self.summax.elements) > 0:
            summax_threshold = max(self.summax.elements.values())
            for sideline_key, sideline_val in self.sideline.items():
                integral_res = self.funcA(sideline_val)
                if (integral_res == 0 or
                   (exp_hashed((self.summax.hash_seed + str(sideline_key[0])
                               + SumMax.KEY_SEP + str(sideline_key[1])))
                    / integral_res) > summax_threshold):
                    del self.sideline[sideline_key]

        # Save Sideline size statistics
        self.max_sideline_size = max(self.max_sideline_size, len(self.sideline))
        current_sideline_distinct = len(set([x[0] for x in
                                             self.sideline.keys()]))
        self.max_distinct_keys_sideline = max(self.max_distinct_keys_sideline,
                                              current_sideline_distinct)

        # Optimization: remove unneeded elements from PPSWOR
        integral_res = self.funcB(self.gamma)
        if integral_res == 0:
            self.ppswor = Ppswor(0)
        elif len(self.summax.elements) > 0:
            summax_threshold = max(self.summax.elements.values()) * self.r
            for ppswor_key, ppswor_val in self.ppswor.elements.items():
                if ppswor_val / integral_res > summax_threshold:
                    del self.ppswor.elements[ppswor_key]

        # Update total size statistics
        current_max_total_size = (len(self.ppswor.elements)
                                  + len(self.summax.elements)
                                  + len(self.sideline))
        self.max_total_size = max(self.max_total_size, current_max_total_size)
        current_total_distinct = len(set(self.ppswor.elements.keys()
                                         + self.summax.elements.keys()
                                         + [x[0] for x in
                                            self.sideline.keys()]))
        self.max_distinct_total_size = max(self.max_distinct_total_size,
                                           current_total_distinct)

    def merge(self, another):
        """
        Merges another sample into this sketch. Currently unimplemented.
        """
        raise Exception("TODO")

    def output_sample(self):
        """
        Returns the output sample of the desired size. The output is a tuple
        (elements, gamma), where elements is a dictionary mapping the sampled
        keys to their score/seed in the PPSWOR sampled, and gamma is the same
        threshold gamma from the sketch description in the paper.

        See Algorithm 5 (in the full version) for details.
        """
        new_summax = SumMax(self.k, self.summax.hash_seed)
        new_summax.merge(self.summax)
        integral_res = self.funcA(self.gamma)
        if integral_res > 0:
            for key, value in self.sideline.items():
                new_summax.process(key, integral_res)
        for x in new_summax.elements.keys():
            new_summax.elements[x] *= self.r
        new_ppswor = BottomK(self.k)
        integral_res = self.funcB(self.gamma)
        if integral_res > 0:
            for key, value in self.ppswor.elements.items():
                new_ppswor.process(key, value / integral_res)
        new_ppswor.merge(new_summax)
        return new_ppswor.elements, self.gamma

def seedCDF(w, t, gamma, r, funcA, funcB):
    """
    Computes the conditional inclusion probability for a key, as in Section 5.4
    in the full version.

    In particular, we compute the probability that the seed of a key with
    frequency w is below a threshold t.

    Parameters:
    w: The frequency of the key.
    t: The threshold (we return the probability of the seed being below t).
    gamma: The parameter gamma from the sampling sketch.
    r: The number of output elements per input element (should be ceil(k/eps)).
    funcA: The function A for the soft concave sublinear function.
            Depends on the inverse complement Laplace transform of the function.
        funcB: The function B for the soft concave sublinear function.
            Depends on the inverse complement Laplace transform of the function.
    """
    p1 = numpy.exp(-1.0 * w * t * funcB(gamma))
    p2_low = (numpy.exp(-1.0 * t * funcA(gamma) / r)
            * (1.0 - numpy.exp(-1.0 * w * gamma)))
    func_to_integr = lambda x: w * numpy.exp(-1.0 * w * x - (t * funcA(x) / r))
    p2_high = integrate.quad(func_to_integr, gamma, numpy.inf)
    p2 = p2_low + p2_high[0]
    return 1.0 - p1 * (p2 ** r)

# The functions A and B (derived from the inverse complement Laplace transform,
# as in Section 5 in the full version) for three soft concave sublinear
# functions: sqrt(x), ln(1 + x), and soft cap with threshold T.

def sqrt_funcA(tau):
    """
    The function A for the soft concave sublinear function sqrt(x).

    Parameters:
    tau: The number on which to evaluate A.
    """
    return (tau * math.pi) ** (-0.5)

def sqrt_funcB(tau):
    """
    The function B for the soft concave sublinear function sqrt(x).

    Parameters:
    tau: The number on which to evaluate B.
    """
    return (tau / math.pi) ** (0.5)

def sqrt_func(x):
    """
    The function sqrt(x).

    Parameters:
    x: The number on which to evaluate the function.
    """
    return math.sqrt(x)

def ln_funcA(tau):
    """
    The function A for the soft concave sublinear function ln(1 + x).

    Parameters:
    tau: The number on which to evaluate A.
    """
    return scipy.special.exp1(tau)

def ln_funcB(tau):
    """
    The function B for the soft concave sublinear function ln(1 + x).

    Parameters:
    tau: The number on which to evaluate B.
    """
    return 1.0 - (math.e ** (-1.0 * tau))

def ln_func(x):
    """
    The function ln(1 + x).

    Parameters:
    x: The number on which to evaluate the function.
    """
    return math.log(1.0 + x, math.e)

def softcap_funcA(tau, T):
    """
    The function A for soft cap with threshold T.

    Parameters:
    tau: The number on which to evaluate A.
    T: The threshold for soft cap.
    """
    if tau <= 1.0/T:
        return T
    return 0.0

def softcap_funcB(tau, T):
    """
    The function B for soft cap with threshold T.

    Parameters:
    tau: The number on which to evaluate B.
    T: The threshold for soft cap.
    """
    if tau <= 1.0/T:
        return 0.0
    return 1.0

def softcap(w, T):
    """
    The soft cap function with threshold T.

    Parameters:
    w: The number on which to evaluate the function.
    T: The threshold for soft cap.
    """
    return T * (1 - numpy.exp(-1.0 * (float(w) / T)))

def estimate(elements, k, func, funcA, funcB, hash_seed="", eps=0.5):
    """
    Produces a sample from a list of data elements and uses it to estimate
    the f-statistics (see Section 2 of the full version) of the dataset.
    These estimates were computed in the experiments described in Section 6 (of
    the full version).

    Parameters:
    elements: The input data elements. An list (or other iterable collection)
        of (key, value) tuples.
    k: The sample size.
    func: The concave sublinear function f (for which to estimate the
        f-statistics).
    funcA: The function A corresponding to f (derived from the inverse
        complement Laplace transform of f).
    funcB: The function B corresponding to f (derived from the inverse
        complement Laplace transform of f).
    hash_seed: A string used to seed the hash function. This determines
            some of the randomness used for sampling and should be different
            across repetitions.
    eps: The parameter epsilon (trades off element processing time with
        variance).
    """
    # First pass over the data (produces the sample).
    sk = MainSketch(k, eps, funcA, funcB, hash_seed)
    for key, value in elements:
        sk.process(key, value)
    output_sample, gamma = sk.output_sample()

    # Determines the inclusion threshold (the k-th lowest seed).
    t = max(output_sample.values())
    # A list of the sampled keys (at most k - 1 keys with lowest seed).
    k_minus_one = [x for x in output_sample.keys() if output_sample[x] < t]
    # Sanity check: there are k - 1 elements with seed below the inclusion
    # threshold
    if len(k_minus_one) != len(output_sample) - 1:
        raise Exception("WARNING: k-1 size is less than k-1")

    # Second pass over the data (gets the frequencies of the sampled keys).
    counts = {}
    for key, value in elements:
        if key in k_minus_one:
            if key not in counts:
                counts[key] = 0
            counts[key] += value
    
    # Computes the inverse probability estimator.
    return sum([func(counts[key])
                / seedCDF(counts[key], t, gamma, sk.r, funcA, funcB)
                for key in k_minus_one])
