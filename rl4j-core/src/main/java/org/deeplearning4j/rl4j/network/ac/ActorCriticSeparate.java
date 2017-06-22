package org.deeplearning4j.rl4j.network.ac;

import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/23/16.
 */
public class ActorCriticSeparate implements IActorCritic {

    final protected MultiLayerNetwork valueNet;
    final protected MultiLayerNetwork policyNet;


    public ActorCriticSeparate(MultiLayerNetwork valueNet, MultiLayerNetwork policyNet) {
        this.valueNet = valueNet;
        this.policyNet = policyNet;
    }

    public void fit(INDArray input, INDArray[] labels) {

        valueNet.fit(input, labels[0]);
        policyNet.fit(input, labels[1]);

    }


    public INDArray[] outputAll(INDArray batch) {
        return new INDArray[] {valueNet.output(batch), policyNet.output(batch)};
    }

    public ActorCriticSeparate clone() {
        return new ActorCriticSeparate(valueNet.clone(), policyNet.clone());
    }

    public Gradient[] gradient(INDArray input, INDArray[] labels) {
        valueNet.setInput(input);
        valueNet.setLabels(labels[0]);
        valueNet.computeGradientAndScore();
        policyNet.setInput(input);
        policyNet.setLabels(labels[1]);
        policyNet.computeGradientAndScore();
        return new Gradient[] {valueNet.gradient(), policyNet.gradient()};
    }


    public void applyGradient(Gradient[] gradient, int batchSize) {
        INDArray g0 = valueNet.getFlattenedGradients();
        g0.assign(gradient[0].gradient());
        valueNet.getUpdater().update(valueNet, new DefaultGradient(g0), 1, batchSize);
        valueNet.params().subi(g0);

        INDArray g1 = policyNet.getFlattenedGradients();
        g1.assign(gradient[1].gradient());
        policyNet.getUpdater().update(policyNet, new DefaultGradient(g1), 1, batchSize);
        policyNet.params().subi(g1);
    }

    public double getLatestScore() {
        return valueNet.score();
    }

    public void save(OutputStream stream) {
        System.out.println("NOT IMPLEMENTED NOOO");
    }

    public void save(String path) {
        System.out.println("NOT IMPLEMENTED NOOO");
    }
}


