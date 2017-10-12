package org.deeplearning4j.rl4j.policy;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

/**
 * An epsilon greedy policy chooses the next action either randomly or passes
 * it on to a policy to decide the next action.
 * <p>
 * Episilon is annealed to minEpsilon over epsilonNbStep steps, so as time progresses it
 * is more likely that the policy (neural net) will be used to determine the next action
 * vs. just randomly picking one.
 *
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/24/16.
 */
@AllArgsConstructor
@Slf4j
public class EpsGreedy<O extends Encodable, A> extends Policy<O, A> {

    final private Policy<O, A> policy;
    final private MDP<?, A, ?> mdp;
    final private int updateStart;
    final private int epsilonNbStep;
    final private Random rd;
    final private float minEpsilon;
    final private StepCountable learning;

    public NeuralNet getNeuralNet() {
        return policy.getNeuralNet();
    }

    public A nextAction(INDArray input) {
        float ep = getEpsilon();
        if (learning.getStepCounter() % 500 == 1)
            log.info("EP: " + ep + " " + learning.getStepCounter());

        if (rd.nextFloat() > ep)
            return policy.nextAction(input);
        else
            return mdp.getActionSpace().randomAction();
    }

    public float getEpsilon() {
        return Math.min(1f, Math.max(minEpsilon, 1f - (learning.getStepCounter() - updateStart) * 1f / epsilonNbStep));
    }
}
