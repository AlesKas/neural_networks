def solution(S):
    solutionNum = 0
    largesSubseq = 0
    # Lenghts of subsequences
    lengths = []
    # Currently processed character
    currentChar = None
    # Counter for number of occurence of char
    counter = 0
    while (len(S) > 0):
        # Get first char
        char = S[0]
        S = S[1:]
        # First, set currentChar to first char of input
        if currentChar is None:
            currentChar = char
        # If next occurence of same char is found, increase counter
        if char == currentChar:
            counter += 1
        else:
            if counter > largesSubseq:
                largesSubseq = counter
            lengths.append(counter)
            currentChar = char
            counter = 1
        if (len(S) == 0):
            if counter > largesSubseq:
                largesSubseq = counter
            lengths.append(counter)

    for length in lengths:
        solutionNum += largesSubseq - length

    return solutionNum

print(solution('babaa'))
print(solution('abba'))
print(solution('ababababab'))
print(solution('aaaaaab'))

a = [1,2,3,5,4]
sorted(a)
print(a)

def solutionB(A):
    # write your code in Python 3.6
    A.sort()
    totalPollution = sum(A)
    result = 0
    filteredPollution = totalPollution
    while (filteredPollution > totalPollution / 2):
        maxPolluter = A[-1]
        maxPolluter /= 2
        A[-1] = maxPolluter
        result += 1
        A.sort()
        filteredPollution = sum(A)


    return result

print(solutionB([5,19,8,1]))