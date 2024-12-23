% Define the base case: the sum of numbers from 1 to 0 is 0.
sum_to_n(0, 0).

% Define the recursive case: the sum of numbers from 1 to N is N + sum of numbers from 1 to N-1.
sum_to_n(N, Sum) :-
    N > 0,
    N1 is N - 1,
    sum_to_n(N1, Sum1),
    Sum is Sum1 + N.
