package org.deeplearning4j.rl4j.mdp;

import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;

public interface HasActionSpace<A> {
    ActionSpace<A> getActionSpace();
}
