from .mixed_strategy_agent import MixedStrategyAgent

rockAgent     = MixedStrategyAgent(support_vector=[1, 0, 0])
paperAgent    = MixedStrategyAgent(support_vector=[0, 1, 0])
scissorsAgent = MixedStrategyAgent(support_vector=[0, 0, 1])
randomAgent   = MixedStrategyAgent(support_vector=[1/3, 1/3, 1/3])
