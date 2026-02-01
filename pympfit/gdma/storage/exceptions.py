"""Exceptions raised when storing GDMA data."""

from openff.recharge.utilities.exceptions import RechargeException


class IncompatibleDBVersion(RechargeException):
    """Exception raised when GDMA store version doesn't match expected version."""

    def __init__(self, found_version: int, expected_version: int) -> None:
        """Initialize the exception.

        Parameters
        ----------
        found_version
            The version of the database being loaded.
        expected_version
            The expected version of the database.
        """
        super().__init__(
            f"The database being loaded is currently at version {found_version} "
            f"while the framework expects a version of {expected_version}. There "
            f"is no way to upgrade the database at this time, although this may "
            f"be added in future versions. Either regenerate the database, or use "
            f"an older compatible version of the `openff-recharge` framework."
        )

        self.found_version = found_version
        self.expected_version = expected_version
