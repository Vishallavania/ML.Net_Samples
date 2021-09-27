using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace AzureAI_SentimentAnalysis
{
    internal class AzureAISentimentPrediction : AzureAISentiment
    {
        [ColumnName("PredictedLabel")] 
        public bool PredictedValue { get; set; }

        public float PredictedProbability { get; set; }

        public float PredictedScore { get; set; }
    }
}