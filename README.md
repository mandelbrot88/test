print("hello")
class HumanAgent(object):
    ''' A human agent for Leduc Holdem. It can be used to play against trained models
    '''

    def __init__(self, num_actions):
        ''' Initilize the human agent
        Args:
            num_actions (int): the size of the ouput action space
        '''
        self.use_raw = True
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Human agent will display the state and make decisions through interfaces
        Args:
            state (dict): A dictionary that represents the current state
        Returns:
            action (int): The action decided by human
        '''
        #_print_state(state['raw_obs'], state['action_record'])
        action = int(input('>> You choose action (integer): '))
        #while action < 0 or action >= len(state['legal_actions']):
        #    print('Action illegel...')
        #    action = int(input('>> Re-choose action (integer): '))
        #return state['raw_legal_actions'][action]
        return action

    def eval_step(self, state):
        ''' Predict the action given the curent state for evaluation. The same to step here.
        Args:
            state (numpy.array): an numpy array that represents the current state
        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state), {}


class CFRAgent(object):
    ''' A human agent for Leduc Holdem. It can be used to play against trained models
    '''

    def __init__(self, game, num_actions):
        ''' Initilize the human agent
        Args:
            num_actions (int): the size of the ouput action space
        '''
        self.use_raw = True
        self.num_actions = num_actions
        self.game = game
        self.path = "/home/user/projects/rebtest/hotest/holder/exps/adhoc/2022-08-03T10:06:36.534425/c02_selfplay/hunl_sp_debug@data.train_batch_size@3_fbe0e7af7d9e7239/ckpt/epoch10.net"
        self.max_depth = int(2)
        self.params = SubgameSolvingParams
        self.seed = 0
        self.root_only = False
        self._beliefs = [
            np.ones(shape=(self.game.num_hands,)) * 1 / self.game.num_hands
            for _ in range(self.game.num_players)
        ]
    @staticmethod
    def step(self, game, state):
        ''' Human agent will display the state and make decisions through interfaces
        Args:
            state (dict): A dictionary that represents the current state
        Returns:
            action (int): The action decided by human
        '''
        net = create_torch_net(self.path, "cpu")
        print("net", net)
        self.params.num_iters = 32
        self.params.linear_update =True
        self.params.dcfr = False
        self.params.max_depth = 2
        this_tree = unroll_tree(game=game, root=state, max_depth=self.max_depth, is_subgame=False)
        print("this_tree len", len(this_tree))
        #strategy = compute_sampled_strategy_recursive_to_leaf(game=game, subgame_params=self.params, net=self.net, seed=self.seed, root_only=self.root_only)
        #print("strategy", strategy)
        random.seed(self.seed)
        # emulate linear weigting: choose only even iterations
        iteration_weights: List[float] = []
        for i in range(self.params.num_iters):
            iteration_weights.append(0. if i % 2 else i / 2 + 1)
        print("iteration_weights", iteration_weights)

        def solver_builder(
                game: Game, node_id: int, state: PartialPublicState, beliefs: Pair
        ) -> ISubgameSolver:
            #act_iteration = sampling(unnormed_probs=iteration_weights)
            #print("act_iteration", act_iteration)
            params = copy(self.params)
            params.num_iters = 2
            if self.root_only and (node_id != 0):
                params.max_depth = int(1e5)
            return build_solver(
                game=game,
                params=params,
                root=state,
                beliefs=beliefs,
                net=net
            )
        print("solver_builder", solver_builder)
        strategy: TreeStrategy = init_nd(
            shape=(len(this_tree), game.num_hands, game.num_actions),
            value=0
        )
        #print("strategy", strategy)
        sample_strategy, new_beliefs = compute_strategy_recursive_to_leaf_for_test(
            game=game,
            tree=this_tree,
            node_id=0,
            beliefs=self._beliefs,
            solver_builder=solver_builder,
            use_samplig_strategy=True,
            p_strategy=strategy
        )
        root_id = 0
        print("sample_strategy", sample_strategy[root_id])
        print("new_beliefs", new_beliefs)
        beliefs = self._beliefs[state.player_id]
        print("beliefs", beliefs)
        hand = sampling(unnormed_probs=beliefs)
        print("hand", hand)
        policy = sample_strategy[root_id, hand]
        action = sampling(unnormed_probs=policy)
        print("action", action)
        return action


    def eval_step(self, state):
        ''' Predict the action given the curent state for evaluation. The same to step here.
        Args:
            state (numpy.array): an numpy array that represents the current state
        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state), {}

human_agent = HumanAgent(9)

game: Game = Game(
            num_hole_cards=2,
            num_deck_cards=52,
            stack_size=200,
            small_blind=1,
            big_blind=2,
            max_raise_times=10,
            raise_ratios=[0.333, 0.5, 0.667, 1, 1.5, 2]
        )
cfr_agent = CFRAgent(game, 9)
state = game.get_initial_state()
print("state", state)


while not game.is_terminal(state):
    print(">> Start a new game")

    #trajectories, payoffs = env.run(is_training=False)
    act = human_agent.step(state)
    new_state = game.act(state, act)
    print("new_state", new_state)
    cfr_act = cfr_agent.step(cfr_agent, game, new_state)
    _state_cfr = game.act(new_state, cfr_act)
    print("_state_cfr", _state_cfr)
    state = _state_cfr
    # If the human does not take the final action, we need to
    # print other players action
    #final_state = trajectories[0][-1]
    #action_record = final_state['action_record']
    #state = final_state['raw_obs']
    #_action_list = []
    #for i in range(1, len(action_record)+1):
    #    if action_record[-i][0] == state['current_player']:
    #        break
    #    _action_list.insert(0, action_record[-i])
    #for pair in _action_list:
    #    print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('===============     CFR Agent    ===============')
    #print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')
    #if payoffs[0] > 0:
    #    print('You win {} chips!'.format(payoffs[0]))
    #elif payoffs[0] == 0:
    #    print('It is a tie.')
    #else:
    #    print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")
