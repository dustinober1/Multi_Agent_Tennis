Code Examples
=============

This section provides practical examples of using the Multi-Agent Tennis codebase.

.. toctree::
   :maxdepth: 2
   
   basic_usage
   custom_training
   evaluation

Basic Usage Examples
--------------------

Training a Model
~~~~~~~~~~~~~~~~

Here's how to train a MADDPG model from scratch:

.. code-block:: python

    from src.maddpg_agent import MADDPGAgent
    from src.config import MADDPGConfig
    
    # Initialize configuration
    config = MADDPGConfig()
    
    # Create agent
    agent = MADDPGAgent(
        state_size=24,
        action_size=2, 
        num_agents=2,
        random_seed=42
    )
    
    # Training loop
    for episode in range(config.max_episodes):
        states = env.reset()
        scores = np.zeros(2)
        
        while True:
            actions = agent.act(states)
            next_states, rewards, dones, _ = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            
            if np.any(dones):
                break

Loading and Evaluating Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.evaluate import ModelEvaluator
    
    # Load trained models
    evaluator = ModelEvaluator(model_path="./models")
    
    # Run evaluation
    results = evaluator.evaluate(episodes=100)
    print(f"Average score: {results['mean_score']:.3f}")

Configuration Examples
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.config import MADDPGConfig
    
    # Custom configuration
    config = MADDPGConfig()
    config.lr_actor = 1e-3
    config.lr_critic = 2e-3
    config.batch_size = 128
    config.noise_sigma = 0.05
    
    # Save configuration
    config.save("my_config.json")

Web Demo Customization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import streamlit as st
    from src.web_demo import TennisDemo
    
    # Create custom demo
    demo = TennisDemo()
    
    # Add custom visualization
    @st.cache_data
    def custom_analysis():
        return demo.analyze_performance()
    
    # Display results
    st.plotly_chart(custom_analysis())

Advanced Examples
-----------------

Custom Training Loop
~~~~~~~~~~~~~~~~~~~~

For more control over the training process:

.. code-block:: python

    import torch
    from src.maddpg_agent import MADDPGAgent
    from src.logger import TrainingLogger
    
    # Initialize components
    agent = MADDPGAgent(24, 2, 2, 42)
    logger = TrainingLogger("experiment_1")
    
    # Custom training with logging
    for episode in range(1000):
        # ... training logic ...
        
        # Log progress
        logger.log_episode(episode, scores, avg_score)
        
        # Custom checkpoint saving
        if episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 
                      f"checkpoint_actor_{episode}.pth")

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.hyperparameter_tuning import GridSearchTuner
    
    # Define parameter grid
    param_grid = {
        'lr_actor': [1e-4, 5e-4, 1e-3],
        'lr_critic': [1e-3, 2e-3, 5e-3],
        'batch_size': [32, 64, 128]
    }
    
    # Run tuning
    tuner = GridSearchTuner(param_grid)
    best_params = tuner.search(episodes=500)
