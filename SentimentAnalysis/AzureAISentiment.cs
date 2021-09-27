using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace AzureAI_SentimentAnalysis
{
    internal class AzureAISentiment
    {
        [LoadColumn(0)] 
        public string InputSentimentText;

        [LoadColumn(1)] [ColumnName("Label")] 
        public bool DetectedSentiment;
    }
}