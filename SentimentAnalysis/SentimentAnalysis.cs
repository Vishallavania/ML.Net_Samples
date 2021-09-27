using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace AzureAI_SentimentAnalysis
{
    public static class SentimentAnalysis
    {
        private static readonly string testfilePath =
            Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");

        public static void RunSentimentAnalysis()
        {
            int ch = 1;
            do
            {
                MLContext mlContext = new MLContext();

                IDataView dataView = mlContext.Data.LoadFromTextFile<AzureAISentiment>(testfilePath, hasHeader: false);

                var trainTestSplit = LoadSentimentData(dataView, mlContext);

                IDataView trainingData = trainTestSplit.TrainSet;
                IDataView testData = trainTestSplit.TestSet;
                ITransformer trainedModel = TrainTheModel(mlContext, trainingData);
                Console.WriteLine("\nPlease enter the sentiment text");
                AzureAISentiment aiSentiment = new AzureAISentiment { InputSentimentText = Console.ReadLine() };

                var resultprediction = predict(mlContext, trainedModel, aiSentiment);

                Console.WriteLine($"\n=============== Single Prediction  ===============\n");
                Console.WriteLine(
                    $"Text: {aiSentiment.InputSentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.PredictedValue) ? "Positive" : "Negative")} Sentiment ");
                Console.WriteLine($"\n================End of Process.Hit 0 to exit and 1 to continue==================================\n");
                ch = Convert.ToInt32(Console.ReadLine());

            } while (ch == 1);
        }
        private static DataOperationsCatalog.TrainTestData LoadSentimentData(IDataView dataView, MLContext mlContext)
        {
            //train test data TrainTestData
            DataOperationsCatalog.TrainTestData splitDataView =
                mlContext.Data.TrainTestSplit(dataView, 0.3);

            return splitDataView;
        }

        private static ITransformer TrainTheModel(MLContext mlContext, IDataView trainingData)
        {
            var dataProcessPipeline =
                mlContext.Transforms.Text.FeaturizeText("Features", nameof(AzureAISentiment.InputSentimentText));

            // STEP 3: Set the training algorithm, then create and config the modelBuilder


            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features");


            var trainingPipeline = dataProcessPipeline.Append(trainer);


            // STEP 4: Train the model fitting to the DataSet


            ITransformer trainedModel = trainingPipeline.Fit(trainingData);
            return trainedModel;
        }

        private static AzureAISentimentPrediction predict(MLContext mlContext, ITransformer trainedModel,
            AzureAISentiment aiSentiment)
        {
            // Create prediction engine related to the loaded trained model
            var predEngine =
                mlContext.Model.CreatePredictionEngine<AzureAISentiment, AzureAISentimentPrediction>(trainedModel);


            // Score
            var resultprediction = predEngine.Predict(aiSentiment);
            return resultprediction;
        }
    }
}
