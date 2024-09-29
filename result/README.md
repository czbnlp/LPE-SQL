# Experiment Results Explanation

This file contains all experiment results. Each line represents the evaluation of one example using different methods. The format of each line is as follows:

vote res: <vote_result>, res1: <result1>, res2: <result2>, res3: <result3>, difficulty: <difficulty_level>


Where:
- **vote res**: The overall result using the cross-consistency method (1 = correct, 0 = incorrect).
- **res1**: The result when using `correct rate = 1` (1 = correct, 0 = incorrect).
- **res2**: The result when using `correct rate = 0.5` (1 = correct, 0 = incorrect).
- **res3**: The result when using `correct rate = 0` (1 = correct, 0 = incorrect).
- **difficulty**: The difficulty level of the current example (`simple` or `moderate`).

All the results are listed in the order of the evaluation set.

## Example
vote res: 1, res1: 1, res2: 0, res3: 1, difficulty: moderate


This means:
- The result using the cross-consistency method is **correct**.
- The result with `correct rate = 1` is **correct**.
- The result with `correct rate = 0.5` is **incorrect**.
- The result with `correct rate = 0` is **correct**.
- The difficulty of the problem is **moderate**.

