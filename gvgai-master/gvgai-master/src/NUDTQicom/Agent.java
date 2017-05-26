package NUDTQicom;

import core.game.StateObservationMulti;
import core.player.AbstractMultiPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;
import tools.Utils;

//import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.TimeoutException;


/**
* Created by QI on 16/05/2017.
* National University of Defense Technology, China
* Email: zhangqiy123@nudt.edu.cn
* <p>
* Respect to Google Java Style Guide:
* https://google.github.io/styleguide/javaguide.html
*/

public class Agent extends AbstractMultiPlayer {

	  private static double GAMMA = 0.95;
	  private static long BREAK_MS = 5;
	  private static int SIMULATION_DEPTH = 15;
	  private static int POPULATION_SIZE = 1;
	  
	  private static final double HUGE_NEGATIVE = -1000000.0;
	  private static final double HUGE_POSITIVE =  1000000.0;

	  private double MUT = (1.0 / SIMULATION_DEPTH);
	  private final int[] N_ACTIONS;

	  private ElapsedCpuTimer timer;

	  private int genome[][][];
	  private final HashMap<Integer, Types.ACTIONS>[] action_mapping;
	  private final HashMap<Types.ACTIONS, Integer>[] r_action_mapping;
	  protected Random randomGenerator;

	  private int id, oppID, no_players;
	  private int hisBest[][];
	 // private int lastDepth;


  /**
   * initialize all variables for the agent
   * @param stateObs Observation of the current state.
   * @param elapsedTimer Timer when the action returned is due.
   * @param playerID ID if this agent
   */
  @SuppressWarnings("unchecked")
public Agent(StateObservationMulti stateObs, ElapsedCpuTimer elapsedTimer, int playerID){
	    id = playerID;
	    no_players = stateObs.getNoPlayers();
	    oppID = (id + 1) % no_players;

	    randomGenerator = new Random();
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

	    initGenome(stateObs);
  }
  
  private void initGenome(StateObservationMulti stateObs) {

	    genome = new int[N_ACTIONS[id]][POPULATION_SIZE][SIMULATION_DEPTH];
	    hisBest = new int[N_ACTIONS[id]][SIMULATION_DEPTH];
	    //lastDepth = 1;

	    // Randomize initial genome
	    for (int i = 0; i < genome.length; i++) {
	      for (int j = 0; j < genome[i].length; j++) {
	        for (int k = 0; k < genome[i][j].length; k++) {	 	    
	        	genome[i][j][k] = randomGenerator.nextInt(N_ACTIONS[id]);
	        	if(0==j){
	        		hisBest[i][k] = genome[i][j][k];
	        	}
	        }
	      }
	    }
	    
  }
  
  double microbial_tournament(int[][] actionGenome, StateObservationMulti stateObs, int playerID, int curMaxDepth, int[] hisRecord) throws TimeoutException{
	    
	    //just select one to change 16/05/17 qi
	    int a = (int) ((POPULATION_SIZE - 1) * randomGenerator.nextDouble());
	    
	    double score_a = simulate(stateObs, actionGenome[a], curMaxDepth);
	
	    // genetic operations, generate one offspring
	    for (int i = 0; i < curMaxDepth; i++) {
	    	hisRecord[i] = actionGenome[a][i];
	    	MUT = (i+1)*GAMMA/curMaxDepth;
	      if (randomGenerator.nextDouble() < MUT){
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

	    if(gameOver && (win == Types.WINNER.PLAYER_LOSES || oppWin == Types.WINNER.PLAYER_WINS))
	      return HUGE_NEGATIVE;

	    if(gameOver && (win == Types.WINNER.PLAYER_WINS || oppWin == Types.WINNER.PLAYER_LOSES))
	      return HUGE_POSITIVE;

	    return rawScore;
	  }
  
  private double simulate(StateObservationMulti stateObs, int[] policy, int curMaxDepth)throws TimeoutException{

	    //System.out.println("depth" + depth);
	    long remaining = timer.remainingTimeMillis();
	    if (remaining < BREAK_MS) {
	      //System.out.println("action:" + policy[0]);
	      //System.out.println("depth:" + curMaxDepth);
	      throw new TimeoutException("Timeout");
	    }

	    int depth = 0;
	    StateObservationMulti newstateObs = stateObs.copy();
	    for (; depth < curMaxDepth; depth++) {
	      Types.ACTIONS[] acts = new Types.ACTIONS[newstateObs.getNoPlayers()];
	      
	      int curMaxactions = N_ACTIONS[id];
	      //curMaxactions = newstateObs.getAvailableActions(id).size();
	      policy[depth] = (policy[depth] < curMaxactions) ? policy[depth] : randomGenerator.nextInt(curMaxactions);
	      acts[id] = this.action_mapping[id].get(policy[depth]);
	      acts[oppID] = this.action_mapping[oppID].get(randomGenerator.nextInt(N_ACTIONS[oppID]));

	      newstateObs.advance(acts);
	      if (newstateObs.isGameOver()) {
	        break;
	      }
	    }

	    double score = Math.pow(GAMMA, depth) * evaluateState(newstateObs, id);
	    return score;
	  }
  
  private Types.ACTIONS microbial(StateObservationMulti stateObs, int maxdepth, int iterations) {

	  //each time new initiation, can modify to exploitation history knowledge
	  for (int j = 0; j < no_players; j++) {
		  N_ACTIONS[j] = stateObs.getAvailableActions(j).size();
	  }
	    double[] maxScores = new double[N_ACTIONS[id]];
	    for (int j = 0; j < maxScores.length; j++) {
		   maxScores[j] = Double.NEGATIVE_INFINITY;
		}
	    
	    //depth will increase gradually in limited times
        int curDepth = 1;//lastDepth;
        outerloop: 	 
	    while(curDepth <= SIMULATION_DEPTH){
	    	for (Types.ACTIONS action : stateObs.getAvailableActions(id)) {
	    		
	    		  Types.ACTIONS[] acts = new Types.ACTIONS[no_players];
	    		  acts[id] = action;
	    		  if(id != oppID){
	    			  acts[oppID] = this.action_mapping[oppID].get(randomGenerator.nextInt(N_ACTIONS[oppID]));
	    		  }
	    		  int int_act = this.r_action_mapping[id].get(acts[id]);
	    		  
	              StateObservationMulti stCopy = stateObs.copy();
	              stCopy.advance(acts);

					//load history information
					for (int k = 0; k < POPULATION_SIZE; k++) {
						for (int j = 0; j < curDepth; j++) {
					    	if(0==k){
					    		genome[int_act][k][j] = hisBest[int_act][j];
					    	}
					    	else{
					    		genome[int_act][k][j] = randomGenerator.nextInt(N_ACTIONS[id]);
					    	}	
						}
					}
	              
	              double score = 0;
	              int localRecord[] = new int[curDepth];
	              for (int i = 0; i < iterations; i++) {	  
			          try {
			        	  score = microbial_tournament(genome[int_act], stCopy, id,curDepth,localRecord) + randomGenerator.nextDouble() * 0.00001;
			            } catch (TimeoutException e) {
			          	  //System.out.println(i);
			              break outerloop;
			            }
			          
			          try {
				          if (score > maxScores[int_act]) {
				             maxScores[int_act] = score;
				             for(int m = 0; m < curDepth; m++){
					             hisBest[int_act][m] = localRecord[m];
				             }
				             //System.out.println("actionsel:" + int_act + score);
				          }
			            } catch (Exception e) {}
		          }
		     }
		    curDepth++;
	    }

	    Types.ACTIONS maxAction = this.action_mapping[id].get(Utils.argmax(maxScores));
	    
	    //shift history information
	    for (int i = 0; i < N_ACTIONS[id]; i++) {
		    for (int j=0; j < SIMULATION_DEPTH-1; j++) {	    
		    	genome[i][0][j] = hisBest[i][j+1];
		    	hisBest[i][j] = genome[i][0][j];
	    	}
		    hisBest[i][SIMULATION_DEPTH-1] =  randomGenerator.nextInt(N_ACTIONS[id]);
	    }
	    //lastDepth = curDepth;
	    
	    return maxAction;
	  }

  /**
   * return ACTION_NIL on every call to simulate doNothing player
   * @param stateObs Observation of the current state.
   * @param elapsedTimer Timer when the action returned is due.
   * @return 	ACTION_NIL all the time
   */
  @Override
  public Types.ACTIONS act(StateObservationMulti stateObs, ElapsedCpuTimer elapsedTimer) {
	    this.timer = elapsedTimer;

	    Types.ACTIONS lastGoodAction = microbial(stateObs, SIMULATION_DEPTH, 8);

	    return lastGoodAction;
  }
}

