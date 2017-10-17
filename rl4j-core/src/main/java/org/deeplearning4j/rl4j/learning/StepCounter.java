package org.deeplearning4j.rl4j.learning;

import lombok.Getter;

/**
 * A simple step counter that can be injected into a Learning object. An object that gets injected
 * into a to be constructed Learning object will not have access to the Learning object as it has not yet been created.
 * To avoid such cycles we inject all the dependencies into the learning tree itself.
 */
public class StepCounter implements StepCountable {

    @Getter
    private int stepCounter = 0;

    public int increment() {
        return this.stepCounter++;
    }
}
