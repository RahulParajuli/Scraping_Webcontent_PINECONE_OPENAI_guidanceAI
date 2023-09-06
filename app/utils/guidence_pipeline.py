
import json
import guidance as gd
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
import openai
import pinecone
from app.utils.openAIprocessor import OpenAIProcessor

from settings import OPENAI_APIKEY, PINECONE_APIKEY, PINECONE_ENV

pinecone.init(api_key= PINECONE_APIKEY, environment=PINECONE_ENV)
openai.api_key = OPENAI_APIKEY

class GuidancePipeline():
    
    def __init__(self, course_information:str="", model:str="") -> None:
        self.course_information = course_information
        self.model = model
        gd.llm = gd.llms.OpenAI("gpt-3.5-turbo-16k") if not self.model else gd.llms.OpenAI(self.model)
    
    def get_guided_prompt(self, html):
        
        guided_prompt = gd("""Parse the details of the {self.course_information} course from the provided html and convert it into a standard json format.
                            Retrivel information such as : Source, Source_MainURL, Course Title, Course_URL, Overview, What You'll Learn, Requirements, Description, Who this Course is for:, Syllabus, Application Deadline, Start Date, Application Process, Duration
                            Program Length Details, Online/Offline, Currency, Price_min, Price_max, Price_tuition_local, Price_tuition_outofstate, Price_tuition_international
                            Price_course, Price_program, Price_raw, Price_details, Price_student_loans, Academics - Curriculum, Academics - Programs Count, Academics - Programs List, Academics - Credits Count, Academics - Courses Count, Academics - Courses List, Language, Instructor, Instructor_Rating, Instructor_RatingCount, Instructor_StudentCount, Instructor_CourseCount, number_students, rating_score, rating_count, rating_reviewcount, LastUpdated, tags, categories
                            \n
                            Then, please create any other new fields that would be useful, and parse the information into those fields that don't already exist.
                            \n
                            HTML : {{await 'HTML'}}
                            JSON Response : {{gen 'updated_response'}}""", stream=True)
        
        return guided_prompt
    
    def get_pinecone_response(self,query, index_name = "udemyrecords"):
        """
        Returns the response from pinecone
        input: query, index_name
        output: contexts
        """
        query = "All course details" if not query else query
        pinecone.init(api_key=PINECONE_APIKEY, environment=PINECONE_ENV)
        index = pinecone.Index(index_name)
        xq = OpenAIProcessor.openAI_embedding(query)
        response = index.query([xq], top_k = 100 ,include_metadata=True)
        contexts = [item['metadata']['HTML'] for item in response['matches']]
        print(len(contexts))
        return contexts

    def get_query_response(self, query:str="") -> str:
        
        final_response = []
        
        response = self.get_pinecone_response(query)
        for i in response:
            guided_prompt = gd("""
                            {{#system~}}
                            You are a helpful html to json parser.
                            {{~/system}}   
                            Parse the details of the {self.course_information} course from the provided html and convert it into a standard json format.
                            Retrive information such as : Source, Source_MainURL, Course Title, Course_URL, Overview, What You'll Learn, Requirements, Description, Who this Course is for:, Syllabus, Application Deadline, Start Date, Application Process, Duration
                            Program Length Details, Online/Offline, Currency, Price_min, Price_max, Price_tuition_local, Price_tuition_outofstate, Price_tuition_international
                            Price_course, Price_program, Price_raw, Price_details, Price_student_loans, Academics - Curriculum, Academics - Programs Count, Academics - Programs List, Academics - Credits Count, Academics - Courses Count, Academics - Courses List, Language, Instructor, Instructor_Rating, Instructor_RatingCount, Instructor_StudentCount, Instructor_CourseCount, number_students, rating_score, rating_count, rating_reviewcount, LastUpdated, tags, categories
                            \n
                            Then, please create any other new fields that would be useful, and parse the information into those fields that don't already exist.
                            \n
                            HTML : {{HTML}}
                            JSON Response : ```json{{gen 'updated_response'}}```""", stream=True)
            guided_prompt(HTML=i)
            print(guided_prompt)
            # response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo-16k",
            #     messages=[
            #         {"role": "system", "content": "You are a helpful html to json parser."},
            #         {"role": "user", "content": updated_query}
            #     ]
            # )
            # print(response['choices'][0]['message']['content'])
            # final_response.append(response['choices'][0]['message']['content'])
            return final_response
    
    
if __name__ == "__main__":
    res = (GuidancePipeline().get_query_response(query="Java"))
    with open("app/utils/response.json", "w") as f:
        json.dump(res, f)