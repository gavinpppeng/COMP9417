# Question
Topic 3.4: Recommender system using collaborative filtering
Task description
Pick one of the the following problems for recommendation and apply collaborative filtering as described in these lecture notes (or an alternative). The data is from a collection collected by the GroupLens research group. Note that there are different versions of this data, with different permissions.
The suggested problems are either: learning to predict the ratings of books on the BookCrossing dataset, or movie ratings on the MovieLens Data Sets (these are of different sizes, so start with the 100K). However, other options are also available from the GroupLens site.
If you want to use one of the larger versions of the MovieLens data, there is one at Kaggle.
Team 2 to 4 person team
Difficulty 5/5


#
SUMMARY & USAGE LICENSE
=============================================
The main.py and evaluation.py should be in the same directory with the dataset.
That means that the u1.base, u2.base, u3.base ... and u1.test, u2.test, u3.test ... and evaluation.py, main.py should be in the same directory.

main.py Including User-base Recommender System and Item-base Recommender System
Step1: calculate the similarity of users(User-base)/movies(Item-base) in training database, and save in a json file, the format is [userid:[similarity, comparing_userid], [similarity, other_comparing_userid]]...For example, '1':[1.0, '941'] means that the user 1 and the user 941 have 1 similarity. In Item-base System, it is the same but the userid is replaced by movieid. Format like: [movieid:[similarity, comparing_movieid], [similarity, other_comparing_movieid]] BTW: The methods of calculating similarity in two system are different, but they are exactly the same as the two methods in PDF. In User-base system, the method is Pearson correlation coefficient(Adjusted Cosine Similarity). In Item-base system, the formula is the same as PDF.
Step2: predict the rating in the test database. Use the simimilarity to predict the rating. Both two system use the same method (Weighted Sum) like PDF. However, some users/movies may never make/be made a score. They are two different situations: 1.If a user doesn't make any scores, I used the average scores of the movies as his predicted scores. 2.If a movie doesn't be made any scores, it is meaningless that calculating the average score of corresponding users. Therefore, I just used the random score(4.0) as the predicted score. (Also in User-base, some users don't have any similar users and the movies that they make scores are also not found in the training dataset(but it exists in test dataset), we used the 3.0 as our predicted score, since we found that the average scores of all dataset about movies are about 3.1, and the user can just mark 3.0 or 3.5, we finally choose the 3.0 as our predicted score.)
Step3: evaluation the result. Use the true value and the predicted value to calculate the RMSE and MAE. The users_result.txt save the predicted score, the format is: userid movieid rating. It is predicted result from User-base System. The movie_result.txt has the same format but it is from Item-base System.

evaluation.py Including Evaluation function which is evaluating our result of these two comparing system. For these two system, there are the same method to evaluate their rating

evaluation.ipynb This is the result of evaluation.py in chart form.

ABOUT DATASET
===============================================
The DATASET is saved in our google drive:
https://drive.google.com/file/d/1feA_qCRQFpIyrWibvgbpMvwy9noHHOK2/view?usp=sharing
The file name is ml-100k.zip, it includes u1.base, u2.base..., u1.test...

MovieLens data sets were collected by the GroupLens Research Project
at the University of Minnesota.
 
This data set consists of:
	* 100,000 ratings (1-5) from 943 users on 1682 movies. 
	* Each user has rated at least 20 movies. 
        * Simple demographic info for the users (age, gender, occupation, zip)

The data was collected through the MovieLens web site
(movielens.umn.edu) during the seven-month period from September 19th, 
1997 through April 22nd, 1998. This data has been cleaned up - users
who had less than 20 ratings or did not have complete demographic
information were removed from this data set. Detailed descriptions of
the data file can be found at the end of this file.

Neither the University of Minnesota nor any of the researchers
involved can guarantee the correctness of the data, its suitability
for any particular purpose, or the validity of results based on the
use of the data set.  The data set may be used for any research
purposes under the following conditions:

     * The user may not state or imply any endorsement from the
       University of Minnesota or the GroupLens Research Group.

     * The user must acknowledge the use of the data set in
       publications resulting from the use of the data set
       (see below for citation information).

     * The user may not redistribute the data without separate
       permission.

     * The user may not use this information for any commercial or
       revenue-bearing purposes without first obtaining permission
       from a faculty member of the GroupLens Research Project at the
       University of Minnesota.

If you have any further questions or comments, please contact GroupLens
<grouplens-info@cs.umn.edu>. 

CITATION
==============================================

To acknowledge use of the dataset in publications, please cite the 
following paper:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent
Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
DOI=http://dx.doi.org/10.1145/2827872


ACKNOWLEDGEMENTS
==============================================

Thanks to Al Borchers for cleaning up this data and writing the
accompanying scripts.

PUBLISHED WORK THAT HAS USED THIS DATASET
==============================================

Herlocker, J., Konstan, J., Borchers, A., Riedl, J.. An Algorithmic
Framework for Performing Collaborative Filtering. Proceedings of the
1999 Conference on Research and Development in Information
Retrieval. Aug. 1999.

FURTHER INFORMATION ABOUT THE GROUPLENS RESEARCH PROJECT
==============================================

The GroupLens Research Project is a research group in the Department
of Computer Science and Engineering at the University of Minnesota.
Members of the GroupLens Research Project are involved in many
research projects related to the fields of information filtering,
collaborative filtering, and recommender systems. The project is lead
by professors John Riedl and Joseph Konstan. The project began to
explore automated collaborative filtering in 1992, but is most well
known for its world wide trial of an automated collaborative filtering
system for Usenet news in 1996.  The technology developed in the
Usenet trial formed the base for the formation of Net Perceptions,
Inc., which was founded by members of GroupLens Research. Since then
the project has expanded its scope to research overall information
filtering solutions, integrating in content-based methods as well as
improving current collaborative filtering technology.

Further information on the GroupLens Research project, including
research publications, can be found at the following web site:
        
        http://www.grouplens.org/

GroupLens Research currently operates a movie recommender based on
collaborative filtering:

        http://www.movielens.org/

DETAILED DESCRIPTIONS OF DATA FILES
==============================================

Here are brief descriptions of the data.

ml-data.tar.gz   -- Compressed tar file.  To rebuild the u data files do this:
                gunzip ml-data.tar.gz
                tar xvf ml-data.tar
                mku.sh

u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of 
	         user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC   

u.info     -- The number of users, items, and ratings in the u data set.

u.item     -- Information about the items (movies); this is a tab separated
              list of
              movie id | movie title | release date | video release date |
              IMDb URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie
              is of that genre, a 0 indicates it is not; movies can be in
              several genres at once.
              The movie ids are the ones used in the u.data data set.

u.genre    -- A list of the genres.

u.user     -- Demographic information about the users; this is a tab
              separated list of
              user id | age | gender | occupation | zip code
              The user ids are the ones used in the u.data data set.

u.occupation -- A list of the occupations.

u1.base    -- The data sets u1.base and u1.test through u5.base and u5.test
u1.test       are 80%/20% splits of the u data into training and test data.
u2.base       Each of u1, ..., u5 have disjoint test sets; this if for
u2.test       5 fold cross validation (where you repeat your experiment
u3.base       with each training and test set and average the results).
u3.test       These data sets can be generated from u.data by mku.sh.
u4.base
u4.test
u5.base
u5.test

ua.base    -- The data sets ua.base, ua.test, ub.base, and ub.test
ua.test       split the u data into a training set and a test set with
ub.base       exactly 10 ratings per user in the test set.  The sets
ub.test       ua.test and ub.test are disjoint.  These data sets can
              be generated from u.data by mku.sh.

allbut.pl  -- The script that generates training and test sets where
              all but n of a users ratings are in the training data.

mku.sh     -- A shell script to generate all the u data sets from u.data.
