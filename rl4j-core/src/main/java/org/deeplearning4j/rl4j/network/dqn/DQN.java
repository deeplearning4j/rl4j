package org.deeplearning4j.rl4j.network.dqn;

import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/25/16.
 */
public class DQN implements IDQN {

    final protected MultiLayerNetwork mln;

    int i = 0;

    public DQN(MultiLayerNetwork mln) {
        this.mln = mln;
    }

    public static DQN load(String path) {
        DQN dqn = null;
        try {
            dqn = new DQN(ModelSerializer.restoreMultiLayerNetwork(path));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dqn;
    }

    public void fit(INDArray input, INDArray labels) {
        mln.fit(input, labels);
    }

    public void fit(INDArray input, INDArray[] labels) {
        fit(input, labels[0]);
    }

    public INDArray output(INDArray batch) {
        return mln.output(batch);
    }

    public INDArray[] outputAll(INDArray batch) {
        return new INDArray[] {output(batch)};
    }

    public DQN clone() {
        return new DQN(mln.clone());
    }

    public Gradient[] gradient(INDArray input, INDArray labels) {
        mln.setInput(input);
        mln.setLabels(labels);
        mln.computeGradientAndScore();
        //System.out.println("SCORE: " + mln.score());
        return new Gradient[] {mln.gradient()};
    }

    public Gradient[] gradient(INDArray input, INDArray[] labels) {
        return gradient(input, labels[0]);
    }

    public void applyGradient(Gradient[] gradient, int batchSize) {
        INDArray g = mln.getFlattenedGradients();
        g.assign(gradient[0].gradient());
        mln.getUpdater().update(mln, new DefaultGradient(g), 1, batchSize);
        mln.params().subi(g);
    }

    public double getLatestScore() {
        return mln.score();
    }

    public void save(OutputStream stream) {
        try {
            ModelSerializer.writeModel(mln, stream, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void save(String path) {
        try {
            ModelSerializer.writeModel(mln, path, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
