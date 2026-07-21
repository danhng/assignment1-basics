import numpy as np
import logging
import linear

# 1. Create a custom logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('main')
# # 2. Create handlers
# c_handler = logging.StreamHandler()  # For console
# # 3. Create formatters and add to handlers
# c_format = logging.Formatter('%(levelname)s - line %(lineno)d - %(message)s')
# c_handler.setFormatter(c_format)
# c_handler.setLevel(logging.DEBUG)
# # 4. Add handlers to the logger
# logger.addHandler(c_handler)

## Usage
if __name__ == '__main__':
    # testTokenizer()
    linear = linear.Linear(10, 20, None, None, None)
    for param in linear.parameters():
        print(param.data)
