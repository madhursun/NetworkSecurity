import sys
from networksecurity.logging.logger import logging

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)

        _, _, exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.error_message = error_message

    def __str__(self):
        return f"Error occurred in script: [{self.file_name}] at line number: [{self.lineno}] with error message: [{str(self.error_message)}]"


if __name__ == "__main__":
    try:
        logging.info("This is an info message")
        a = 1 / 0
    except Exception as e:
        raise NetworkSecurityException(e, sys)