import numpy as np

UTILITY_VALUE = .000001


class RandomWalk:

    def __init__(self, alphas, lambdas):
        self.alphas = alphas
        self.lambdas = lambdas
        self.true_probs = [1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6]
        self.results = []
        self.dtype = np.float

    def train_repeated(self, training_sets):
        for _lambda in self.lambdas:
            for alpha in self.alphas:
                rmses = []
                for training_set in training_sets:
                    # values initialized to zero and updates via tdlEstimate
                    values = np.zeros(7, dtype=self.dtype)
                    iterations = 0

                    while True:
                        iterations += 1
                        before = np.copy(values)
                        updates = np.zeros(7, dtype=self.dtype)
                        values[6] = 1.0  # Reward for right-side termination

                        for sequence in training_set:
                            updates += self._get_tdl_estimate(alpha,
                                                              _lambda,
                                                              sequence, values)

                        values += updates
                        diff = np.sum(np.absolute(before - values))

                        if diff < UTILITY_VALUE:
                            break

                    estimate = np.array(values[1:-1], dtype=self.dtype)
                    rmses.append(self._get_rms(estimate))

                result = self._build_result_row(_lambda, alpha, rmses)
                self.results.append(result)

        return self.results

    def train_single(self, training_sets):
        results = []
        for _lambda in self.lambdas:
            for alpha in self.alphas:
                rms_vals = []
                for training_set in training_sets:

                    values = np.array([0.5 for i in range(7)])

                    for sequence in training_set:
                        values[0] = 0.0
                        values[6] = 1.0
                        values += self._get_tdl_estimate(alpha, _lambda,
                            sequence, values)

                    estimate = np.array(values[1:-1])
                    error = (self.true_probs - estimate)
                    rms   = np.sqrt(np.average(np.power(error, 2)))

                    rms_vals.append(rms)

                result = [_lambda, alpha, np.mean(rms_vals), np.std(rms_vals)]
                results.append(result)
        return results


    def _get_tdl_estimate(self, alpha, _lambda, state_sequence, values):
        """Compute TD Lambda Estimate."""
        eligibility = np.zeros(7)
        updates = np.zeros(7)

        for t in range(0, len(state_sequence) - 1):
            current_state = state_sequence[t]
            next_state = state_sequence[t+1]

            eligibility[current_state] += 1.0

            td = alpha * (values[next_state] - values[current_state])

            updates += td * eligibility
            eligibility *= _lambda

        return updates

    def _get_rms(self, estimate):
        """Calculates the RMS error for a given estimate."""
        error = (self.true_probs - estimate)
        rms = np.sqrt(np.average(np.power(error, 2)))
        return rms

    def _build_result_row(self, _lambda, alpha, rmses):
        return [_lambda, alpha, np.mean(rmses), np.std(rmses)]
