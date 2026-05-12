# exception/custom_exception.py
# Custom exception class for the product assistant.
# Captures the full traceback, file name, and line number of where the error
# originated, making it easy to log structured error information.

import sys
import traceback
from typing import Optional, cast

class ProductAssistantException(Exception):
    """Rich exception that captures file, line, message, and full traceback.
    
    Usage:
        raise ProductAssistantException("Something failed", sys)
        raise ProductAssistantException("Something failed", original_exception)
    """

    def __init__(self, error_message, error_details: Optional[object] = None):
        # Normalize the error message to a string regardless of input type
        if isinstance(error_message, BaseException):
            norm_msg = str(error_message)
        else:
            norm_msg = str(error_message)

        # Resolve exception info from various sources:
        # - None: grab current exception from sys.exc_info()
        # - sys module: call sys.exc_info() explicitly
        # - Exception object: extract type, value, and traceback directly
        exc_type = exc_value = exc_tb = None
        if error_details is None:
            # No details provided — try to get the current exception context
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            if hasattr(error_details, "exc_info"):  # e.g., the sys module itself
                exc_info_obj = cast(sys, error_details)
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
            elif isinstance(error_details, BaseException):
                # An exception object was passed directly
                exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__
            else:
                # Fallback: try to get info from the current exception context
                exc_type, exc_value, exc_tb = sys.exc_info()

        # Walk the traceback chain to the last (deepest) frame
        # This gives us the most relevant file/line where the error actually occurred
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        # Extract file name and line number from the deepest traceback frame
        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        # Format the full traceback as a string for logging
        if exc_type and exc_tb:
            self.traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self):
        # Compact, logger-friendly format: file, line, message + optional traceback
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        return f"ProductAssistantException(file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"


# if __name__ == "__main__":
#     # Demo-1: generic exception -> wrap
#     try:
#         a = 1 / 0
#     except Exception as e:
#         raise ProductAssistantException("Division failed", e) from e

#     # Demo-2: still supports sys (old pattern)
#     # try:
#     #     a = int("abc")
#     # except Exception as e:
#     #     raise ProductAssistantException(e, sys)