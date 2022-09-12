from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional


@dataclass
class LossTracker:

    sensor_loss: float = field(default=-1.0)
    l2_sensor_loss: float = field(default=-1.0)
    momentum_loss: float = field(default=-1.0)
    continuity_loss: float = field(default=-1.0)
    total_loss: float = field(default=-1.0)
    l2_actual_loss: float = field(default=-1.0)

    def get_dict(self, training: Optional[bool] = None) -> dict[str, Any]:

        """Get loss dictionary.

        Parameters
        ----------
        training: bool
            Whether these losses were generated in the training phase.

        Returns
        -------
        dict[str, float]
            dictionary representing the results.
        """

        return {f'{self._prepend_str(k, training)}': v for k, v in asdict(self).items()}

    def get_fields(self, training: Optional[bool] = None) -> list[str]:

        """Get field names.

        Parameters
        ----------
        training: bool
            Whether these losses were generated in the training phase.

        Returns
        -------
        list[str]
            List of the field names defined above.
        """

        return list(map(lambda x: self._prepend_str(x.name, training), filter(lambda x: x.repr, fields(self))))

    @property
    def get_loss_keys(self) -> list[float]:

        """Get loss values.

        Returns
        -------
        list[float]
            List of the loss values corresponding to the fields above.
        """

        return list(map(lambda x: getattr(self, x.name), fields(self)))

    @staticmethod
    def _prepend_str(msg: str, training: Optional[bool]) -> str:

        """Prepend training / validation string.

        Parameters
        ----------
        msg: str
            String to prepend message to.
        training: bool
            Whether this message corresponds to the training phase.

        Returns
        -------
        prepended_msg: str
            Message prepended with '', 'train_', 'validation_'
        """

        ps = ''
        if training is not None:
            ps = 'train_' if training else 'validation_'

        prepended_msg = f'{ps}{msg}'

        return prepended_msg
