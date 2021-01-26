import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class IrisApp {
    public static void main(String[] args) throws Exception {

        int batchSize=1;
        int outputSize=3;
        int classIndex=4;
        double learninRate=0.001;
        int inputSize=4;
        int numHiddenNodes=10;
        int nEpochs=100;


        /* ************************************************
          Etape 1: la création et la configuration de modèle
         ************************************************* */

        // La configuration du modèle
        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(learninRate))
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(numHiddenNodes)
                        .activation(Activation.SIGMOID).build())
                .layer(1,new OutputLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(outputSize)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .activation(Activation.SOFTMAX).build())
                .build();
        // Affichage de la configuration
        System.out.println("=== La configuration ===");
        System.out.println(configuration.toJson());
        // La création du modèle
        MultiLayerNetwork model=new MultiLayerNetwork(configuration);
        // Inisialisation du modèle
        model.init();

         /* **************************
          Etape 2: http://localhost:9000 DL4J Training UI
         **************************** */

        UIServer uiServer = UIServer.getInstance();
        InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        model.setListeners(new StatsListener(inMemoryStatsStorage));

        /* *******************************
          Etape 3: Entrainement du modèle
         ********************************* */
        System.out.println(" Entrainement du modèle");

        File fileTrain = new ClassPathResource("iris-train.csv").getFile();
        RecordReader recordReaderTrain = new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain,batchSize,classIndex,outputSize);
        while (dataSetIteratorTrain.hasNext()){
            System.out.println("===================");
            DataSet dataSet = dataSetIteratorTrain.next();
            System.out.println(dataSet.getFeatures());
            System.out.println(dataSet.getLabels());
        }
        for (int i = 0; i <nEpochs ; i++) {
            model.fit(dataSetIteratorTrain);
        }

         /* *****************************
          Etape 4: Évaluation le modèle
         ******************************** */
        System.out.println(" Évaluation le modèle ");

        File fileTest = new ClassPathResource("irisTest.csv").getFile();
        RecordReader recordReaderTest = new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest,batchSize,classIndex,outputSize);
        Evaluation evaluation = new Evaluation(outputSize);

        while (dataSetIteratorTest.hasNext()){
            DataSet dataSet = dataSetIteratorTest.next();
            INDArray features=dataSet.getFeatures();
            INDArray labelss=dataSet.getLabels();
            INDArray predicted=model.output(features);
            evaluation.eval(labelss,predicted);
        }
        System.out.println(evaluation.stats());

         /* ************************************
          Etape 5: Sauvegarde le modèle
         *************************************** */
        System.out.println("Sauvegarde le modèle");

        ModelSerializer.writeModel(model,new File("model.zip"),true);
    }
}
