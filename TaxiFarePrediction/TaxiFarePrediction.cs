using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace AzureAI_SentimentAnalysis
{
    public static class TaxiFarePrediction
    {
        private static readonly string datasetfilePath =
             Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare.csv");

        public static void RunTaxiFarePrediction()
        {
            int ch = 1;
            do
            {
                MLContext mlContext = new MLContext();

                IDataView dataView = mlContext.Data.LoadFromTextFile<AzureAIFareData>(datasetfilePath, hasHeader: true, separatorChar: ',');
                var count = dataView.GetColumn<float>(nameof(AzureAIFareData.FareAmount)).Count();
                IDataView trainingDataView = mlContext.Data.FilterRowsByColumn(dataView, nameof(AzureAIFareData.FareAmount), lowerBound: 1, upperBound: 150);
                var count2 = trainingDataView.GetColumn<float>(nameof(AzureAIFareData.FareAmount)).Count();
                var trainTestSplit = LoadIssueData(dataView, mlContext);

                IDataView trainingData = trainTestSplit.TrainSet;
                IDataView testData = trainTestSplit.TestSet;

                ITransformer trainedModel = TrainTheModel(mlContext, trainingData);

                //Console.WriteLine("\nPlease enter the sentiment text");
                var taxiTripSample = new AzureAIFareData()
                {
                    VendorId = "VTS",
                    RateCode = "1",
                    PassengerCount = 1,
                    TripTime = 1140,
                    TripDistance = 3.75f,
                    PaymentType = "CRD",
                    FareAmount = 0 // To predict. Actual/Observed = 15.5
                };

                var resultprediction = predict(mlContext, trainedModel, taxiTripSample);

                Console.WriteLine($"\n=============== Single Prediction  ===============\n");
                Console.WriteLine($"VendorId: {taxiTripSample.VendorId} | RateCode:{taxiTripSample.RateCode}\n" +
                    $"PassengerCount:{taxiTripSample.PassengerCount} | TripTime:{taxiTripSample.TripTime}\n" +
                    $"TripDistance:{taxiTripSample.TripDistance} | PaymentType:{taxiTripSample.PaymentType}\n" +
                    $"FareAmount:{taxiTripSample.FareAmount}\n");
                Console.WriteLine($"Predicted fare: {resultprediction.FareAmount:0.####}, actual fare: 15.5");
                Console.WriteLine($"\n================End of Process.Hit 0 to exit and 1 to continue==================================\n");

                ch = Convert.ToInt32(Console.ReadLine());

            } while (ch == 1);
        }

        private static AzureAIFarePrediction predict(MLContext mlContext, ITransformer trainedModel, AzureAIFareData aiFareData)
        {
            var predEngine = mlContext.Model.CreatePredictionEngine<AzureAIFareData, AzureAIFarePrediction>(trainedModel);

            // Prediction
            var resultprediction = predEngine.Predict(aiFareData);
            
            return resultprediction;
        }

        private static ITransformer TrainTheModel(MLContext mlContext, IDataView trainingData)
        {
            var dataProcessPipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(AzureAIFareData.FareAmount))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: nameof(AzureAIFareData.VendorId)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: nameof(AzureAIFareData.RateCode)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: nameof(AzureAIFareData.PaymentType)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(AzureAIFareData.PassengerCount)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(AzureAIFareData.TripTime)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(AzureAIFareData.TripDistance)))
                            .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded", nameof(AzureAIFareData.PassengerCount)
                            , nameof(AzureAIFareData.TripTime), nameof(AzureAIFareData.TripDistance)));

            // STEP 3: Set the training algorithm, then create and config the modelBuilder

            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

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
