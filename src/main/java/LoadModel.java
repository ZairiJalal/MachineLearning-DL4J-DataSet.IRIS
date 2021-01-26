import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class LoadModel {

    public static void main(String[] args) throws Exception {
        String[] labels={"Iris-setosa","Iris-versicolor","Iris-virginica"};
        System.out.println("Loading Model");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new
                File("model.zip"));
        System.out.println("Prédiction");
        INDArray input= Nd4j.create(new double[][]{
                {6.9,3.1,5.4,2.1}

        });
        INDArray output=model.output(input);
        int classIndex =output.argMax(1).getInt(0);
        System.out.println(labels[classIndex]);

    }
}
