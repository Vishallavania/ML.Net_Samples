using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace AzureAI_SentimentAnalysis
{
    public static class IssueClassification
    {
        private static readonly string datasetfilePath =
             Path.Combine(Environment.CurrentDirectory, "Data", "corefx_issues.tsv");

        public static void RunIssueClassification()
        {
            int ch = 1;
            do
            {
                MLContext mlContext = new MLContext();

                IDataView dataView = mlContext.Data.LoadFromTextFile<AzureAI_Issue>(datasetfilePath, hasHeader: true, separatorChar: '\t', allowSparse: false);

                var trainTestSplit = LoadIssueData(dataView, mlContext);

                IDataView trainingData = trainTestSplit.TrainSet;
                IDataView testData = trainTestSplit.TestSet;

                ITransformer trainedModel = TrainTheModel(mlContext, trainingData);

                //Console.WriteLine("\nPlease enter the sentiment text");
                var aiIssue = new AzureAI_Issue
                {
                    ID = "25",
                    Title = "WebSockets communication is slow in my machine",
                    Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
                };

                var resultprediction = predict(mlContext, trainedModel, aiIssue);

                Console.WriteLine($"\n=============== Single Prediction  ===============\n");
                Console.WriteLine($"Title: {aiIssue.Title}\nDescription: {aiIssue.Description} \n\n" +
                    $"Prediction-Result: {resultprediction.Area} ");
                Console.WriteLine($"\n================End of Process.Hit 0 to exit and 1 to continue==================================\n");
                
                ch = Convert.ToInt32(Console.ReadLine());

            } while (ch == 1);
        }

        private static AzureAI_IssuePrediction predict(MLContext mlContext, ITransformer trainedModel, AzureAI_Issue aiIssue)
        {
            var predEngine = mlContext.Model.CreatePredictionEngine<AzureAI_Issue, AzureAI_IssuePrediction>(trainedModel);

            // Prediction
            var resultprediction = predEngine.Predict(aiIssue);
            return resultprediction;
        }

        private static ITransformer TrainTheModel(MLContext mlContext, IDataView trainingData)
        {
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(AzureAI_Issue.Area))
                            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "TitleFeaturized", inputColumnName: nameof(AzureAI_Issue.Title)))
                            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "DescriptionFeaturized", inputColumnName: nameof(AzureAI_Issue.Description)))
                            .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "TitleFeaturized", "DescriptionFeaturized"))
                            .AppendCacheCheckpoint(mlContext);

            // STEP 3: Set the training algorithm, then create and config the modelBuilder

            IEstimator<ITransformer> trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: Train the model fitting to the DataSet

            ITransformer trainedModel = trainingPipeline.Fit(trainingData);
            return trainedModel;
        }

        private static DataOperationsCatalog.TrainTestData LoadIssueData(IDataView dataView, MLContext mlContext)
        {
            //train test data TrainTestData
            DataOperationsCatalog.TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, 0.2);

            return splitDataView;
        }
    }
}
