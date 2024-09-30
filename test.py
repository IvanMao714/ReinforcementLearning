# def find_longest_good_subarray(financial_metrics, limit):
#     n = len(financial_metrics)
#     max_length = -1
#
#
#     for right in range(n):
#         left = 0
#         # Calculate the length of the current window
#         length = right - left + 1
#         # Calculate the threshold for the current window length
#         threshold = limit // length
#
#         # Adjust the left pointer to ensure all elements in the window are greater than the threshold
#         while left <= right and financial_metrics[right] <= threshold:
#             left += 1
#             length = right - left + 1
#             if length > 0:
#                 threshold = limit // length
#
#         # Update the maximum length of the good subarray found
#         if left <= right and financial_metrics[right] > threshold:
#             max_length = max(max_length, right - left + 1)
#
#     return max_length if max_length > 0 else -1
def find_longest_good_subarray(financial_metrics, limit):
    n = len(financial_metrics)
    max_length = 0

    for start in range(n):
        for end in range(start, n):
            subarray_length = end - start + 1
            is_good = True

            for i in range(start, end + 1):
                if financial_metrics[i] <= limit / subarray_length:
                    is_good = False
                    break

            if is_good:
                print(start, end, subarray_length)
                max_length = max(max_length, subarray_length)

    return max_length if max_length > 0 else -1

# Example usage
financialMetrics = [1, 3, 4, 3, 1]
limit = 6
print(find_longest_good_subarray(financialMetrics, limit))  # Example output: 3

def min_additional_cards(cardTypes):
    maxCards = max(cardTypes)
    additional_cards_needed = sum(maxCards - cards for cards in cardTypes)
    return additional_cards_needed

# 示例使用：
cardTypes = [3, 5, 2, 7, 4]
print(min_additional_cards(cardTypes))