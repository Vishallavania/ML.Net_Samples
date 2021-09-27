using System;
using System.Data;
using System.IO;
using Microsoft.ML;

namespace AzureAI_SentimentAnalysis
{
    public class Program
    {
        private static readonly string testfilePath =
            Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");

        private static void Main(string[] args)
        {
            int choice;
            do 
            {                
                Console.WriteLine($"\n=============== Welcome to ML.Net Lab ===============\n");
                Console.WriteLine($"1) Sentiment Analysis\n2) IssueClassification\n3) Taxi Fare Prediction");                    
                Console.WriteLine($"\n================End of Menu...Hit 0 to exit========================\n");
                choice = Convert.ToInt32(Console.ReadLine());
                
                switch(choice)
                {
                    case 1: SentimentAnalysis.RunSentimentAnalysis();
                            break;

                    case 2: IssueClassification.RunIssueClassification();
                            break;

                    case 3: TaxiFarePrediction.RunTaxiFarePrediction();
                            break;
                    
                    default: Console.WriteLine("Please provide valid input\n");
                             break;
                }
            } while (choice > 0 && choice < 4);
        }       
    }
}