import os
import datetime
import inspect
import traceback

LOG_DIR = "log_files"

"""
Class for the logging functionality.
"""
class Logger:
    # Log levels
    INFO = 20
    ERROR = 40

    LEVEL_NAMES = {
        INFO: "INFO",
        ERROR: "ERROR"
    }

    def __init__(self, log_name):
        """
        Initialize the logger with a log file name.

        Args:
            log_name (str): Name of the log file (without extension)
        """
        # Get the directory where logger.py is located
        logger_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the log_files directory within the log directory
        self.log_dir = os.path.join(logger_dir, "log_files")
        self.log_name = log_name
        self.log_file = os.path.join(self.log_dir, f"{log_name}.log")

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

    def _write_to_log(self, level, message, module=None, exc_info=None):
        """
        Write a message to the log file.

        Args:
            level (int): Message log level (INFO or ERROR)
            message (str): Message to log
            module (str, optional): Module name where the log was generated
            exc_info (Exception, optional): Exception information to include
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = self.LEVEL_NAMES.get(level, "UNKNOWN")

        if module is None:
            # Get the caller's frame
            frame = inspect.currentframe().f_back.f_back

            # Get the file name (including path)
            file_path = frame.f_code.co_filename if frame else "unknown"
            file_name = os.path.basename(file_path)

            # Get the function name and line number
            function_name = frame.f_code.co_name if frame else "unknown"
            line_number = frame.f_lineno if frame else 0

            # Format the caller information
            caller = f"{file_name}:{function_name}:{line_number}"
        else:
            caller = module

        # Format the log entry
        log_entry = f"[{timestamp}] {level_name} - {caller}: {message}"

        # Add exception info if provided
        if exc_info:
            log_entry += f"\n{traceback.format_exc()}"

        # Append to the log file
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")

    def info(self, message):
        """Log an info message."""
        self._write_to_log(self.INFO, message)

    def error(self, message, exc_info=None):
        """
        Log an error message with optional exception information.

        Args:
            message (str): Error message
            exc_info (bool/Exception, optional): If True, includes current exception info
        """
        self._write_to_log(self.ERROR, message, exc_info=exc_info)

    def exception(self, message):
        """
        Log an exception with traceback (convenience method).
        Equivalent to error(message, exc_info=True)
        """
        self._write_to_log(self.ERROR, message, exc_info=True)

    def clear_log(self):
        """Clear the content of the log file."""
        with open(self.log_file, "w") as f:
            f.write("")

    def get_log_path(self):
        """Return the full path to the log file."""
        return os.path.abspath(self.log_file)
