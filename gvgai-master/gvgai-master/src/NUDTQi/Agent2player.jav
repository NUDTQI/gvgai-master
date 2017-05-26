package NUDTQi;

import core.game.StateObservationMulti;
import core.player.AbstractMultiPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;
import tools.Utils;
import tools.Vector2d;

import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.TimeoutException;
import java.util.ArrayDeque;

/**
 * Created by QI on 16/05/2017. National University of Defense Technology, China
 * Email: zhangqiy123@nudt.edu.cn
 * <p>
 * Respect to Google Java Style Guide:
 * https://google.github.io/styleguide/javaguide.html
 */

public class Agent extends AbstractMultiPlayer {

	private static double GAMMA = 0.99;
	private static long BREAK_MS = 5;
	private static int SIMULATION_DEPTH = 10;
	private static int POPULATION_SIZE = 1;

	private static final double HUGE_NEGATIVE = -1000000.0;
	private static final double HUGE_POSITIVE = 1000000.0;

	private double MUT = 0.5;
	private final int[] N_ACTIONS;

	private ElapsedCpuTimer timer;

	private int genome[][][];
	private final HashMap<Integer, Types.ACTIONS> action_mapping[];
	private final HashMap<Types.ACTIONS, Integer> r_action_mapping[];
	protected Random randomGenerator;

	private int id, oppID, no_players;
	private int[][] hisBest;
	
	//get heuristics
	private int trainingTicks = 20;
	private int testTicks = 10;
	private int curTick = 0;
	
	private ArrayDeque<Vector2d>  trainSetPos; //x,y
	private ArrayDeque<Vector2d> testSetPos; //x,y
	//private ArrayDeque<Vector2d> gravitySetPos; //x,y
	private Vector2d BaryCenter;
	private Vector2d AccumulatedOffset;
	private double deltaAverOffset = 0;
	
	private ArrayDeque<Double> rewardSet; //x,y
	private double curScore = 0;
	private double lastScore = 0;
	private double sumScoreIntervel = 0;


	/**
	 * initialize all variables for the agent
	 * 
	 * @param stateObs
	 *            Observation of the current state.
	 * @param elapsedTimer
	 *            Timer when the action returned is due.
	 * @param playerID
	 *            ID if this agent
	 */
	@SuppressWarnings("unchecked")
	public Agent(StateObservationMulti stateObs, ElapsedCpuTimer elapsedTimer, int playerID) {
		id = playerID;
		no_players = stateObs.getNoPlayers();
		oppID = (id + 1) % no_players;

		N_ACTIONS = new int[no_players];
		action_mapping = new HashMap[no_players];
		r_action_mapping = new HashMap[no_players];
	    for (int j = 0; j < no_players; j++) {
	        action_mapping[j] = new HashMap<>();
	        r_action_mapping[j] = new HashMap<>();
	        int i = 0;
	        for (Types.ACTIONS action : stateObs.getAvailableActions(j)) {
	          action_mapping[j].put(i, action);
	          r_action_mapping[j].put(action, i);
	          i++;
	        }
	        N_ACTIONS[j] = stateObs.getAvailableActions(j).size();
	      }
	    
	    trainSetPos = new ArrayDeque<Vector2d>();
	    testSetPos = new ArrayDeque<Vector2d>();
	    //gravitySetPos = new ArrayDeque<Vector2d>();
	    rewardSet = new ArrayDeque<Double>();
	    BaryCenter = new Vector2d();
	    AccumulatedOffset = new Vector2d();
	    
		randomGenerator = new Random();
		initGenome(stateObs);
	}

	private void initGenome(StateObservationMulti stateObs) {

		genome = new int[N_ACTIONS[id]][POPULATION_SIZE][SIMULATION_DEPTH];
		hisBest = new int[N_ACTIONS[id]][SIMULATION_DEPTH];

		// Randomize initial genome
		for (int i = 0; i < genome.length; i++) {
			for (int j = 0; j < genome[i].length; j++) {
				for (int k = 0; k < genome[i][j].length; k++) {
					genome[i][j][k] = randomGenerator.nextInt(N_ACTIONS[id]);
					if (0 == j) {
						hisBest[i][k] = genome[i][j][k];
					}
				}
			}
		}

	}

	double microbial_tournament(int[][] actionGenome, StateObservationMulti stateObs, int playerID, int curMaxDepth,
			int[] hisRecord) throws TimeoutException {

		// just select one to change 16/05/17 qi
		int a = (int) ((POPULATION_SIZE - 1) * randomGenerator.nextDouble());

		double score_a = simulate(stateObs, actionGenome[a], curMaxDepth);

		// genetic operations, generate one offspring
		for (int i = 0; i < curMaxDepth; i++) {
			hisRecord[i] = actionGenome[a][i];
			MUT = (i + 1) * GAMMA / curMaxDepth;
			if (randomGenerator.nextDouble() < MUT) {
				actionGenome[a][i] = randomGenerator.nextInt(N_ACTIONS[playerID]);
			}
		}

		return score_a;
	}

	public double evaluateState(StateObservationMulti stateObs, int playerID) {
		boolean gameOver = stateObs.isGameOver();
		Types.WINNER win = stateObs.getMultiGameWinner()[playerID];
		Types.WINNER oppWin = stateObs.getMultiGameWinner()[(playerID + 1) % stateObs.getNoPlayers()];
		double rawScore = stateObs.getGameScore(playerID);

		if (gameOver && (win == Types.WINNER.PLAYER_LOSES || oppWin == Types.WINNER.PLAYER_WINS))
			return HUGE_NEGATIVE;

		if (gameOver && (win == Types.WINNER.PLAYER_WINS || oppWin == Types.WINNER.PLAYER_LOSES))
			return HUGE_POSITIVE;

		return rawScore-this.curScore;
	}

	private double simulate(StateObservationMulti stateObs, int[] policy, int curMaxDepth) throws TimeoutException {

		long remaining = timer.remainingTimeMillis();
		if (remaining < BREAK_MS) {
			//System.out.println("depth:" + curMaxDepth);
			throw new TimeoutException("Timeout");
		}

		int depth = 0;
		StateObservationMulti newstateObs = stateObs.copy();
		for (; depth < curMaxDepth; depth++) {
			Types.ACTIONS[] actsim = new Types.ACTIONS[newstateObs.getNoPlayers()];

			policy[depth] = (policy[depth] < N_ACTIONS[id]) ? policy[depth] : randomGenerator.nextInt(N_ACTIONS[id]);
			actsim[id] = this.action_mapping[id].get(policy[depth]);
			if(oppID != id){
				actsim[oppID] = this.action_mapping[id].get(randomGenerator.nextInt(N_ACTIONS[oppID]));				
			}

			newstateObs.advance(actsim);
			if (newstateObs.isGameOver()) {
				break;
			}
		}

		double score = Math.pow(GAMMA, depth) * evaluateState(newstateObs, id);
		return score;
	}

	// bandit UCB select
	private int chooseActionUCB(double[] qtable, int totalTimes, int[] visitTimes) {

		double qbestValue = Double.NEGATIVE_INFINITY;
		int bestAction = 0;
		double cPara = 1;
		
		//normalising is essential
		double maxscore = Double.NEGATIVE_INFINITY;
		double minscore = Double.POSITIVE_INFINITY;
		for(int i=0;i<qtable.length;i++){
			if (qtable[i] > maxscore) {
				maxscore = qtable[i];
			}
			if (qtable[i] < minscore) {
				minscore = qtable[i];
			}
		}	

		for (int i = 0; i < qtable.length; i++) {
			double ucbValue;
			if (0 == visitTimes[i]) {
				ucbValue = Double.POSITIVE_INFINITY;
			} else {
				ucbValue = Utils.normalise(qtable[i], minscore, maxscore) + cPara * Math.sqrt(2 * Math.log(totalTimes) / visitTimes[i]) 
				+ randomGenerator.nextDouble() * 0.00001;
			}

			if (ucbValue > qbestValue) {
				qbestValue = ucbValue;
				bestAction = i;
			}
		}

		return bestAction;
	}

	
	private Types.ACTIONS microbial(StateObservationMulti stateObs, int maxdepth) {

		// each time new initiation
		for (int j = 0; j < no_players; j++) {
			N_ACTIONS[j] = stateObs.getAvailableActions(j).size();
		}

		double[] maxScores = new double[N_ACTIONS[id]];
		int[] curDepth = new int[N_ACTIONS[id]];
		int[] visitTimes = new int[N_ACTIONS[id]];
		int totalTimes = 0;
		
		for (int j = 0; j < maxScores.length; j++) {
			maxScores[j] = HUGE_NEGATIVE;
			curDepth[j] = 1;
			visitTimes[j] = 0;
		}
		
		Types.ACTIONS[] acts = new Types.ACTIONS[no_players];
		int int_act = 0;
		
//		Vector2d curPos = stateObs.getAvatarPosition();
//		Vector2d[] nextPos = new Vector2d[N_ACTIONS[id]];
//		double moveDisStep = 10.0;
//		double diffScore = 0;

		// depth will increase gradually in limited times
		outerloop: while (curDepth[int_act] <= SIMULATION_DEPTH) {

			// select action according to UCB bandit
			int_act = chooseActionUCB(maxScores, totalTimes, visitTimes);
			acts[id] = this.action_mapping[id].get(int_act);
			if (id != oppID) {
				acts[oppID] = this.action_mapping[oppID].get(randomGenerator.nextInt(N_ACTIONS[oppID]));
			}
			//System.out.println(int_act + "/" +  totalTimes + "/" +  visitTimes[int_act] + "/" + acts[id]);
			
			StateObservationMulti stCopy = stateObs.copy();
			stCopy.advance(acts);
			
//			if(nextPos[int_act]==null){
//				nextPos[int_act] = stCopy.getAvatarPosition();
//				double temp = curPos.dist(nextPos[int_act]);
//				if(moveDisStep < temp){
//					moveDisStep = temp;
//				}
//				//System.out.println("action:"+int_act + " nextx:" + nextPos[int_act].x + " nexty:" + nextPos[int_act].y);
//			}
			
			// load history information to initial history best
			for (int k = 0; k < POPULATION_SIZE; k++) {
				for (int j = 0; j < curDepth[int_act]; j++) {
					if (0 == k) {
						genome[int_act][k][j] = hisBest[int_act][j];
					} else {
						genome[int_act][k][j] = randomGenerator.nextInt(N_ACTIONS[id]);
					}
				}
			}

			// simulation and search pso
			double score = 0;
			maxScores[int_act] = Double.NEGATIVE_INFINITY;
			int localRecord[] = new int[curDepth[int_act]];
			int iterations = N_ACTIONS[id] + curDepth[int_act];
			
			for (int i = 0; i < iterations; i++) {
				try {
					score = microbial_tournament(genome[int_act], stCopy, id, curDepth[int_act], localRecord)
							+ randomGenerator.nextDouble() * 0.00001;
				} catch (TimeoutException e) {
					break outerloop;
				}

				try {
					if (score > maxScores[int_act]) {
						maxScores[int_act] = score;
						for (int m = 0; m < curDepth[int_act]; m++) {
							hisBest[int_act][m] = localRecord[m];
						}
					}
//					if(Math.abs(score) > diffScore){
//						diffScore = Math.abs(score);
//					}
				} catch (Exception e) {
				}
			}

			curDepth[int_act]++;
			visitTimes[int_act]++;
			totalTimes++;

			// System.out.println("select action:" + int_act);
		}

		int bestActionID = Utils.argmax(maxScores);
		
//		double eps = 0.0001;
//		if(this.curTick > this.trainingTicks+this.testTicks){
//			if(sumScoreIntervel<eps && deltaAverOffset<=curDepth[bestActionID]*moveDisStep && diffScore<eps){
//				double bestdis = Double.NEGATIVE_INFINITY;				
//				for (int j = 0; j < N_ACTIONS[id]; j++) {
//					if(nextPos[j] != null){
//						double dis = nextPos[j].dist(this.BaryCenter) + randomGenerator.nextDouble() * 0.0001;
//						if(dis > bestdis){
//							bestdis = dis;
//							bestActionID = j;
//						}
//					}
//				}
//				//System.out.println("in escape:");
//			}
//		}
		
		Types.ACTIONS maxAction;
		maxAction = stateObs.getAvailableActions(id).get(bestActionID);//this.action_mapping[id].get(bestActionID);
		//System.out.println(bestActionID + "-" + maxAction);
		
		// shift history information
		//int maxDepth = -100;
		for (int i = 0; i < N_ACTIONS[id]; i++) {
			for (int j = 0; j < SIMULATION_DEPTH - 1; j++) {
				genome[i][0][j] = hisBest[i][j + 1];
				hisBest[i][j] = genome[i][0][j];
			}
			hisBest[i][SIMULATION_DEPTH - 1] = randomGenerator.nextInt(N_ACTIONS[id]);

			//if (curDepth[i] > maxDepth) {
			//	maxDepth = curDepth[i];
			//}
		}
		//System.out.println("maxdepth:" + maxDepth);
		return maxAction;
	}
	
	private void UpdateHerustics(StateObservationMulti stateObs, int playerID){
		
		curTick++;
		
		Vector2d curPos = stateObs.getAvatarPosition(playerID);
			
		if(curTick <= trainingTicks){
			trainSetPos.offer(curPos);
			BaryCenter = BaryCenter.add((curPos.x-BaryCenter.x)/curTick,(curPos.y-BaryCenter.y)/curTick);
			
			double deltascore = curScore - lastScore;
			rewardSet.add(deltascore);
			sumScoreIntervel = sumScoreIntervel + deltascore;
		}
		else if(curTick <= trainingTicks+testTicks){
			testSetPos.offer(curPos);
			AccumulatedOffset.add(curPos.x-BaryCenter.x,curPos.y-BaryCenter.y);

			double deltascore = curScore - lastScore;
			rewardSet.offer(deltascore);
			sumScoreIntervel = sumScoreIntervel + deltascore;
		}
		else{
			Vector2d temp = trainSetPos.poll();
			Vector2d temp2 = testSetPos.poll();
			trainSetPos.offer(temp2);
			testSetPos.offer(curPos);
			
			//update latest trainSet center of gravity
			Vector2d deltaBaryCenter = new Vector2d();
			deltaBaryCenter.set((temp2.x-temp.x)/trainingTicks,(temp2.y-temp.y)/trainingTicks);
			BaryCenter = BaryCenter.add(deltaBaryCenter);
			
			AccumulatedOffset.add(curPos.x-temp2.x-testTicks*deltaBaryCenter.x,curPos.y-temp2.y-testTicks*deltaBaryCenter.y);
			
			double deltascore = curScore - lastScore;
			rewardSet.offer(deltascore);
			sumScoreIntervel = sumScoreIntervel - rewardSet.poll() + deltascore;
		}

		deltaAverOffset = AccumulatedOffset.mag();
		
		System.out.println(curTick + ":" + BaryCenter.x + ":" + BaryCenter.y);
	}

	/**
	 * return ACTION_NIL on every call to simulate doNothing player
	 * 
	 * @param stateObs
	 *            Observation of the current state.
	 * @param elapsedTimer
	 *            Timer when the action returned is due.
	 * @return ACTION_NIL all the time
	 */
	@Override
	public Types.ACTIONS act(StateObservationMulti stateObs, ElapsedCpuTimer elapsedTimer) {
		this.timer = elapsedTimer;
		
		this.curScore = stateObs.getGameScore(id);
		
		//this.UpdateHerustics(stateObs, id);

		Types.ACTIONS lastGoodAction = microbial(stateObs, SIMULATION_DEPTH);	
		
		this.lastScore = this.curScore;

		return lastGoodAction;
	}
}
