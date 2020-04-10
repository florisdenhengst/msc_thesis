def check_for_trigram(new_token, trigrams):
        # print(len(trigrams))
        if len(trigrams) == 0:
                trigrams.append([new_token])  
        elif len(trigrams[0]) < 3:
            trigrams[0] = trigrams[0] + [new_token]
        else:
            # assert 1 == 2
            trigrams.append(trigrams[-1][1:] + [new_token])
            print(trigrams.count(trigrams[-1]))
            if trigrams.count(trigrams[-1]) > 1:
                return trigrams, False
        return trigrams, True

if __name__ == '__main__':
    text = "@entity2 's @entity7 was initially deployed in 2004 to help in reconstruction and left in 2006 . @entity2 's military has airlifted materials , troops between @entity12 and @entity5 since 2006 . @entity2 's military has airlifted materials , troops between @entity12 and @entity5 since 2006 . @entity2 's military has airlifted materials , troops between @entity12 and @entity5 since 2006"
    trigrams = []
    for token in text.split(' '):
        trigrams, status = check_for_trigram(token, trigrams)
        print(trigrams[-1])
        print(status)