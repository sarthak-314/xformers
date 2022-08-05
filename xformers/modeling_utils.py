from flax.linen import partitioning as nn_partitioning
with_sharding_constraint = nn_partitioning.with_sharding_constraint
remat = nn_partitioning.remat

ACT2FN = {

}

class ModelConfig:
    """
    Configuration class for the model
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)