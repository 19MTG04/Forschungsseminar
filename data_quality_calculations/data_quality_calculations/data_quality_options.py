from enum import Enum, auto
from dataclasses import dataclass

from accuracy_calculations.accuracy_calculation_options import AccuracyCalculationOptions

from believability_calculations.believability_options import BelievabilityOptions

from objectivity_calculations.objectivity_options import ObjectivityCalculationOptions

from reputation_calculations.reputation_options import ReputationOptions


@dataclass
class CategoryWeightsDataQuality:
    weight_accuracy: float = 1
    weight_believability: float = 1
    weight_objectivity: float = 1
    weight_reputation: float = 1

    def sum_weights(self) -> float:
        return sum(self.__dict__.values())


@dataclass
class DataQualityOptions:
    accuracy_options: AccuracyCalculationOptions
    believability_options: BelievabilityOptions
    objectivity_options: ObjectivityCalculationOptions
    reputation_options: ReputationOptions


class AutoName(Enum):

    # Overrides the auto() function to automatically set the value of the enum
    # variable to the string name.
    @staticmethod
    def _generate_next_value_(name: str, start, count, last_values):
        return name.lower()


class ModelType(str, AutoName):
    FULL_MODEL = auto()
    MODEL_WO_COMMUNITY = auto()
