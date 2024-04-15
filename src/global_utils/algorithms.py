def find_factors_with_minimal_sum(number):
    if number == 1:
        return (1, 1)

    # Initialize variables to keep track of the factors with the minimal sum
    min_sum = float("inf")
    min_factors = None

    # Iterate through potential factors from 1 to half of the number
    for factor1 in range(1, number // 2 + 1):
        factor2 = number // factor1

        # Check if factor1 * factor2 is equal to the original number
        if factor1 * factor2 == number:
            current_sum = factor1 + factor2

            # Update the minimum sum and factors if the current sum is smaller
            if current_sum < min_sum:
                min_sum = current_sum
                min_factors = (factor1, factor2)

    return min_factors