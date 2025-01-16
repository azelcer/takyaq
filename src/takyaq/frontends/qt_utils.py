"""Utility classes for PyQt.

@author: azelcer
"""
from PyQt5.QtWidgets import (QCheckBox, QDoubleSpinBox)


class GroupedCheckBoxes:
    """Manages grouped CheckBoxes states.

    This is a helper class to ease GUI implementation. It implements an 'All'
    checkbox that controls and stays sinchronized with others. It should be used
    carefully to avoid state changes loops.
    """

    def __init__(self, all_checkbox: QCheckBox, *other_checkboxes):
        """Init class.

        Parameters
        ----------
        all_checkbox : QCheckBox
            The checkbox that checks/unchecks all others.
        *other_checkboxes : Iterable[QCheckBox]
            All other checkboxes.
        """
        self.acb = all_checkbox
        self.others = other_checkboxes
        for _ in other_checkboxes:
            _.stateChanged.connect(self.on_state)
        all_checkbox.clicked.connect(self.on_click)

    def on_state(self, state: int):
        """Handle single items change."""
        self.acb.setChecked(all([_.isChecked() for _ in self.others]))

    def on_click(self, is_checked: bool):
        """Handle 'All' checkbox click."""
        for _ in self.others:
            _.setChecked(is_checked)


def create_spin(value: float, decimals: int, step: float,
                minimum: float = 0., maximum: float = 1.):
    """Create spin with properties."""
    rv = QDoubleSpinBox()
    rv.setValue(value)
    rv.setDecimals(decimals)
    rv.setMinimum(minimum)
    rv.setMaximum(maximum)
    rv.setSingleStep(step)
    return rv
