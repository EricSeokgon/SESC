import logging  # Make sure logging is imported

# Configure logging (optional but good practice)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def weighted_mean(num_list, weights):
    running_total = 0  # Indent this line
    for i in range(len(num_list)):
        running_total += (
            num_list[i] * weights[0]
        )  # Indent this line; also fixed index for weights
        logging.debug(
            f"The running total at step {i} is {running_total}"
        )  # Indent this line
    return running_total / len(num_list)


# Example usage:
print(weighted_mean([1, 6, 8], [1, 3, 2]))
