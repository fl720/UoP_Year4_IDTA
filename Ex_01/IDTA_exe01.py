# # PYTHON BASIC

# """
# This 
# allows 
# multi 
# lines 
# comments. 
# """

# for i in range(5 ) :
#     print(i) 

# # exe1
# print("I love python") 

# # exe2
# for i in range(5) :
#     print( "I love php")

# # exe3
# def two_num_func() :
#     fir_num = input("Enter first number:") 
#     sec_num = input("Enter second number:") 
#     return float(fir_num) , float(sec_num)
# fir_num , sec_num = two_num_func()
# two_sum = fir_num+sec_num
# print( f"The total is {two_sum}" )

# # exe4
# fir_num , sec_num = two_num_func()
# if (fir_num * sec_num) < 1000 :
#     two_sum = fir_num + sec_num
#     print(f"The result is: {two_sum}")
# else :
#     two_pro =  fir_num * sec_num 
#     print(f"The result is: {two_pro}")

# exe5
lists = [5,12,17,20,21,25,27,30]
target_mode = 5
print( f"Numbers divisible by {target_mode} from the list {lists} are:")
for i in lists :
    if i%target_mode == 0  :
        print( i )

# exe6
target = 6
for i in range(1,target,1) : 
    for j in range(i) :
        print(i , end=" " )
    print()

# exe7
city_list = ["London" , "Manchester" , "Portsmouth" , "Southampton"]
print( city_list[0] )
print( city_list[-1])

# exe8 
print( city_list[1:3] ) 

# exe9 
tar_para = 5 
for i in range(1, tar_para+1 ) :
    for j in range(1, tar_para+1 ) :
        print( i*j , end=" " ) 
    print( )

# exe10 
def find_word_freq( sentence , tar ) : 
    ans = 0 
    ele = sentence.split( ) 
    for e in ele : 
        if e == tar : 
            ans += 1 
    print( f"The word '{tar}' appeared {ans} time(s).")

sen = "The longest river in the world is the Nile."
find_word_freq( sen , "the" )