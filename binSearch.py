class Struct(): pass

template = Struct()
template.distanceIndices = range(10)

def findIndices(templateDistance):
    if (templateDistance > template.distanceIndices[-1] or
        templateDistance < template.distanceIndices[0]):
        return False
    start = 0
    end = len(template.distanceIndices)
    while True:
        mid = (start + end) / 2
        if template.distanceIndices[mid] == templateDistance:
            return mid - 1, mid + 1
        elif start == end:
            if templateDistance.distanceIndices[start] < templateDistance:
                return start, start + 1
            else:
                return start - 1, start
        elif abs(start - end) == 1:
            return (min(start, end), max(start, end))
        elif template.distanceIndices[mid] < templateDistance:
            start = mid
        else:
            end = mid