using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace AzureAI_SentimentAnalysis
{
    public class AzureAIFarePrediction 
    {
        [ColumnName("Score")] 
        public float FareAmount; 
    }
}
