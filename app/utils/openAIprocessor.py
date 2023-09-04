import os
import openai

openai.api_key = os.getenv('OPENAI_APIKEY')

class OpenAIProcessor():
    """
    This class is used to process the data for openai

    input: None

    parameters: None

    """
    def __init__(self):
        pass
    
    def openAI_embedding(content, engine = "text-embedding-ada-002"):
        """
        input: content, engine
        output: vectors
        """
        content = content.encode(encoding='UTF-8',errors='ignore').decode()
        response = openai.Embedding.create(input = content, engine = engine)
        vectors = response['data'][0]['embedding']
        return vectors
    
    def chatCompletion(content):
        """
        input: content
        output: OpenAI response
        """
        primer = f"""
            Prompt:
            I have the following course information:
            *Course Information*
            ---
            Please parse the course information into the following fields:
            Source
            Source_MainURL
            Course Title
            Course_URL
            Overview
            What You'll Learn
            Requirements
            Description
            Who this Course is for:
            Syllabus
            Application Deadline
            Start Date
            Application Process
            Duration
            Program Length Details
            Online/Offline
            Currency
            Price_min
            Price_max
            Price_tuition_local
            Price_tuition_outofstate
            Price_tuition_international
            Price_course
            Price_program
            Price_raw
            Price_details
            Price_student_loans
            Academics - Curriculum
            Academics - Programs Count
            Academics - Programs List
            Academics - Credits Count
            Academics - Courses Count
            Academics - Courses List
            Language
            Instructor
            Instructor_Rating
            Instructor_RatingCount
            Instructor_StudentCount
            Instructor_CourseCount
            number_students
            rating_score
            rating_count
            rating_reviewcount
            LastUpdated
            tags
            categories
            ---
            Then, please create any other new fields that would be useful, and parse the information into those fields that don't already exist.
    """

        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            temperature = 1,
            messages = [
                {"role": "system", "content": primer},
                {"role": "user", "content": content}
            ]
        )
        return response['choices'][0]['message']['content']