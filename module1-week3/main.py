###### CAU HOI TU LUAN #######
# Exercise 1: Viết class và cài phương thức softmax
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x)
        return exp_x / sum_exp_x


class SoftmaxStable(nn.Module):
    def __init__(self):
        super(SoftmaxStable, self).__init__()

    def forward(self, x):
        c = torch.max(x)
        exp_x = torch.exp(x - c)
        sum_exp_x = torch.sum(exp_x)
        return exp_x / sum_exp_x


# Example usage
data = torch.Tensor([1, 2, 3])

# Using Softmax
softmax = Softmax()
output = softmax(data)
print(output)  # tensor([0.0900, 0.2447, 0.6652])

# Using SoftmaxStable
softmax_stable = SoftmaxStable()
output_stable = softmax_stable(data)
print(output_stable)  # tensor([0.0900, 0.2447, 0.6652])

# Exercise 2: Cài đặt các class Student, Doctor, và Teacher trong một Ward (phường)


class Person:
    def __init__(self, name, yob):
        self.name = name
        self.yob = yob

    def describe(self):
        raise NotImplementedError("Subclass must implement abstract method")


class Student(Person):
    def __init__(self, name, yob, grade):
        super().__init__(name, yob)
        self.grade = grade

    def describe(self):
        print(
            f"Student - Name: {self.name} - YoB: {self.yob} - Grade: {self.grade}")


class Teacher(Person):
    def __init__(self, name, yob, subject):
        super().__init__(name, yob)
        self.subject = subject

    def describe(self):
        print(
            f"Teacher - Name: {self.name} - YoB: {self.yob} - Subject: {self.subject}")


class Doctor(Person):
    def __init__(self, name, yob, specialist):
        super().__init__(name, yob)
        self.specialist = specialist

    def describe(self):
        print(
            f"Doctor - Name: {self.name} - YoB: {self.yob} - Specialist: {self.specialist}")


class Ward:
    def __init__(self, name):
        self.name = name
        self.people = []

    def add_person(self, person):
        self.people.append(person)

    def describe(self):
        print(f"Ward Name: {self.name}")
        for person in self.people:
            person.describe()

    def count_doctor(self):
        return sum(isinstance(person, Doctor) for person in self.people)

    def sort_age(self):
        self.people.sort(key=lambda person: person.yob)

    def compute_average(self):
        teacher_yobs = [
            person.yob for person in self.people if isinstance(person, Teacher)]
        if teacher_yobs:
            return sum(teacher_yobs) / len(teacher_yobs)
        return None


# Creating instances
student1 = Student(name="studentA", yob=2010, grade="7")
teacher1 = Teacher(name="teacherA", yob=1969, subject="Math")
doctor1 = Doctor(name="doctorA", yob=1945, specialist="Endocrinologists")

# Describing instances
student1.describe()
teacher1.describe()
doctor1.describe()

# Creating a Ward and adding people
ward1 = Ward(name="Ward1")
teacher2 = Teacher(name="teacherB", yob=1995, subject="History")
doctor2 = Doctor(name="doctorB", yob=1975, specialist="Cardiologists")

ward1.add_person(student1)
ward1.add_person(teacher1)
ward1.add_person(teacher2)
ward1.add_person(doctor1)
ward1.add_person(doctor2)

# Describing Ward
ward1.describe()

# Counting Doctors
print(f"\nNumber of doctors: {ward1.count_doctor()}")

# Sorting by age and describing Ward
print("\nAfter sorting Age of Ward1 people")
ward1.sort_age()
ward1.describe()

# Computing average year of birth of teachers
print(f"\nAverage year of birth (teachers): {ward1.compute_average()}")

# Exercise 3: Xây dựng class Stack


class MyStack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.stack = []

    def is_empty(self):
        return len(self.stack) == 0

    def is_full(self):
        return len(self.stack) == self.capacity

    def push(self, value):
        if not self.is_full():
            self.stack.append(value)
        else:
            raise Exception("Stack is full")

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            raise Exception("Stack is empty")

    def top(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            raise Exception("Stack is empty")


# Example usage:
stack1 = MyStack(capacity=5)
stack1.push(1)
stack1.push(2)
print(stack1.is_full())   # False
print(stack1.top())       # 2
print(stack1.pop())       # 2
print(stack1.top())       # 1
stack1.push(3)
print(stack1.top())       # 3
print(stack1.pop())       # 3
print(stack1.pop())       # 1
print(stack1.is_empty())  # True

# Exercise 4: Xây dựng class Queue


class MyQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = []
        self.front_index = 0
        self.rear_index = -1

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.capacity

    def enqueue(self, value):
        if not self.is_full():
            self.queue.append(value)
            self.rear_index += 1
        else:
            raise Exception("Queue is full")

    def dequeue(self):
        if not self.is_empty():
            value = self.queue.pop(0)
            self.rear_index -= 1
            return value
        else:
            raise Exception("Queue is empty")

    def front(self):
        if not self.is_empty():
            return self.queue[0]
        else:
            raise Exception("Queue is empty")


# Example usage:
queue1 = MyQueue(capacity=5)
queue1.enqueue(1)
queue1.enqueue(2)
print(queue1.is_full())   # False
print(queue1.front())     # 1
print(queue1.dequeue())   # 1
print(queue1.front())     # 2
queue1.enqueue(3)
print(queue1.front())     # 2
print(queue1.dequeue())   # 2
print(queue1.dequeue())   # 3
print(queue1.is_empty())  # True

###### CAU HOI TRAC NGHIEM #######
# Exercise 1:

data = torch.Tensor([1, 2, 3])
softmax_function = nn.Softmax(dim=0)
output = softmax_function(data)
assert round(output[0].item(), 2) == 0.09
print(output)  # Output: tensor([0.0900, 0.2447, 0.6652])

# Exercise 2:


class MySoftmax(nn.Module):
    def __init__(self):
        super(MySoftmax, self).__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x)
        return exp_x / sum_exp_x


data = torch.Tensor([5, 2, 4])
my_softmax = MySoftmax()
output = my_softmax(data)
assert round(output[-1].item(), 2) == 0.26
print(output)  # Output: tensor([0.7054, 0.0351, 0.2595])

# Exercise 3:


class MySoftmax(nn.Module):
    def __init__(self):
        super(MySoftmax, self).__init__()

    def forward(self, x):
        c = torch.max(x)
        exp_x = torch.exp(x - c)
        sum_exp_x = torch.sum(exp_x)
        return exp_x / sum_exp_x


# Example usage:
data = torch.Tensor([1, 2, 3000000000])
my_softmax = MySoftmax()
output = my_softmax(data)
assert round(output[0].item(), 2) == 0.0
print(output)  # Expected output should be close to [0.0, 0.0, 1.0]

# Exercise 4:


class SoftmaxStable(nn.Module):
    def __init__(self):
        super(SoftmaxStable, self).__init__()

    def forward(self, x):
        x_max = torch.max(x, dim=0, keepdim=True)
        x_exp = torch.exp(x - x_max.values)
        partition = x_exp.sum(dim=0, keepdim=True)
        return x_exp / partition


# Example usage:
data = torch.Tensor([1, 2, 3])
softmax_stable = SoftmaxStable()
output = softmax_stable(data)
assert round(output[-1].item(), 2) == 0.67
print(output)

# Exercise 5:


class Person(ABC):
    def __init__(self, name: str, yob: int):
        self._name = name
        self._yob = yob

    def get_yob(self):
        return self._yob

    @abstractmethod
    def describe(self):
        pass


class Student(Person):
    def __init__(self, name: str, yob: int, grade: str):
        super().__init__(name, yob)
        self._grade = grade

    def describe(self):
        print(
            f"Student - Name: {self._name} - YoB: {self._yob} - Grade: {self._grade}")


# Example usage
student1 = Student(name="studentZ2023", yob=2011, grade="6")
assert student1._yob == 2011
student1.describe()

# Exercise 6:


class Person(ABC):
    def __init__(self, name: str, yob: int):
        self._name = name
        self._yob = yob

    def get_yob(self):
        return self._yob

    @abstractmethod
    def describe(self):
        pass


class Teacher(Person):
    def __init__(self, name: str, yob: int, subject: str):
        super().__init__(name, yob)
        self._subject = subject

    def describe(self):
        print(
            f"Teacher - Name: {self._name} - YoB: {self._yob} - Subject: {self._subject}")


# Example usage
teacher1 = Teacher(name="teacherZ2023", yob=1991, subject="History")
assert teacher1._yob == 1991
teacher1.describe()

# Exercise 7:


class Person(ABC):
    def __init__(self, name: str, yob: int):
        self._name = name
        self._yob = yob

    def get_yob(self):
        return self._yob

    @abstractmethod
    def describe(self):
        pass


class Doctor(Person):
    def __init__(self, name: str, yob: int, specialist: str):
        super().__init__(name, yob)
        self._specialist = specialist

    def describe(self):
        print(
            f"Doctor - Name: {self._name} - YoB: {self._yob} - Specialist: {self._specialist}")


# Example usage
doctor1 = Doctor(name="doctorZ2023", yob=1981, specialist="Endocrinologists")
assert doctor1._yob == 1981
doctor1.describe()

# Exercise 8:


class Ward:
    def __init__(self, name: str):
        self._name = name
        self._list_people = []

    def add_person(self, person):
        self._list_people.append(person)

    def describe(self):
        print(f"Ward Name: {self._name}")
        for p in self._list_people:
            p.describe()

    def count_doctor(self):
        count = 0
        for p in self._list_people:
            if isinstance(p, Doctor):
                count += 1
        return count

# Assuming the classes Student, Teacher, and Doctor are already defined as in previous exercises


class Person(ABC):
    def __init__(self, name: str, yob: int):
        self._name = name
        self._yob = yob

    def get_yob(self):
        return self._yob

    @abstractmethod
    def describe(self):
        pass


class Student(Person):
    def __init__(self, name: str, yob: int, grade: str):
        super().__init__(name, yob)
        self._grade = grade

    def describe(self):
        print(
            f"Student - Name: {self._name} - YoB: {self._yob} - Grade: {self._grade}")


class Teacher(Person):
    def __init__(self, name: str, yob: int, subject: str):
        super().__init__(name, yob)
        self._subject = subject

    def describe(self):
        print(
            f"Teacher - Name: {self._name} - YoB: {self._yob} - Subject: {self._subject}")


class Doctor(Person):
    def __init__(self, name: str, yob: int, specialist: str):
        super().__init__(name, yob)
        self._specialist = specialist

    def describe(self):
        print(
            f"Doctor - Name: {self._name} - YoB: {self._yob} - Specialist: {self._specialist}")


# Example usage:
student1 = Student(name="studentA", yob=2010, grade="7")
teacher1 = Teacher(name="teacherA", yob=1969, subject="Math")
teacher2 = Teacher(name="teacherB", yob=1995, subject="History")
doctor1 = Doctor(name="doctorA", yob=1945, specialist="Endocrinologists")
doctor2 = Doctor(name="doctorB", yob=1975, specialist="Cardiologists")

ward1 = Ward(name="Ward1")
ward1.add_person(student1)
ward1.add_person(teacher1)
ward1.add_person(teacher2)
ward1.add_person(doctor1)
ward1.add_person(doctor2)

ward1.describe()
print(ward1.count_doctor())

# Exercise 9:


class MyStack:
    def __init__(self, capacity):
        self._capacity = capacity
        self._stack = []

    def is_full(self):
        return len(self._stack) == self._capacity

    def push(self, value):
        if not self.is_full():
            self._stack.append(value)
        else:
            raise Exception("Stack is full")


# Example usage:
stack1 = MyStack(capacity=5)
stack1.push(1)
assert stack1.is_full() == False
stack1.push(2)
print(stack1.is_full())

# Exercise 10:


class MyStack:
    def __init__(self, capacity):
        self._capacity = capacity
        self._stack = []

    def is_full(self):
        return len(self._stack) == self._capacity

    def is_empty(self):
        return len(self._stack) == 0

    def push(self, value):
        if not self.is_full():
            self._stack.append(value)
        else:
            raise Exception("Stack is full")

    def top(self):
        if not self.is_empty():
            return self._stack[-1]
        else:
            raise Exception("Stack is empty")


# Example usage:
stack1 = MyStack(capacity=5)
stack1.push(1)
stack1.push(2)
assert stack1.is_full() == False
stack1.push(3)
print(stack1.top())  # Expected output: 3

# Exercise 11:


class MyQueue:
    def __init__(self, capacity):
        self._capacity = capacity
        self._queue = []

    def is_full(self):
        return len(self._queue) == self._capacity

    def enqueue(self, value):
        if not self.is_full():
            self._queue.append(value)
        else:
            raise Exception("Queue is full")


# Example usage:
queue1 = MyQueue(capacity=5)
queue1.enqueue(1)
queue1.enqueue(2)
assert queue1.is_full() == False
queue1.enqueue(3)
print(queue1.is_full())  # Expected output: False

# Exercise 12:


class MyQueue:
    def __init__(self, capacity):
        self._capacity = capacity
        self._queue = []

    def is_empty(self):
        return len(self._queue) == 0

    def is_full(self):
        return len(self._queue) == self._capacity

    def enqueue(self, value):
        if not self.is_full():
            self._queue.append(value)
        else:
            raise Exception("Queue is full")

    def dequeue(self):
        if not self.is_empty():
            return self._queue.pop(0)
        else:
            raise Exception("Queue is empty")

    def front(self):
        if not self.is_empty():
            return self._queue[0]
        else:
            raise Exception("Queue is empty")


# Example usage:
queue1 = MyQueue(capacity=5)
queue1.enqueue(1)
queue1.enqueue(2)
assert queue1.is_full() == False
queue1.enqueue(3)
print(queue1.front())  # Expected output: 1
