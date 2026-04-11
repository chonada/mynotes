# Object-Oriented Programming Concepts in Python
# Comprehensive examples of the four main OOP pillars

print("="*70)
print("OBJECT-ORIENTED PROGRAMMING CONCEPTS IN PYTHON")
print("="*70)

# =============================================================================
# 1. ENCAPSULATION
# =============================================================================

print("\n1. ENCAPSULATION")
print("-" * 40)
print("Encapsulation bundles data and methods together and restricts direct access")
print("to internal implementation details.\n")

class BankAccount:
    """
    Demonstrates encapsulation by hiding internal data and providing
    controlled access through methods
    """
    
    def __init__(self, account_number, initial_balance=0):
        # Public attribute
        self.account_number = account_number
        
        # Protected attribute (convention: single underscore)
        self._account_type = "Savings"
        
        # Private attribute (name mangling: double underscore)
        self.__balance = initial_balance
        self.__transaction_history = []
        
        # Add initial transaction
        self.__add_transaction("Account opened", initial_balance)
    
    # Private method (internal use only)
    def __add_transaction(self, description, amount):
        """Private method to record transactions"""
        transaction = {
            'description': description,
            'amount': amount,
            'balance': self.__balance
        }
        self.__transaction_history.append(transaction)
    
    # Public methods (interface to interact with the object)
    def deposit(self, amount):
        """Public method to deposit money"""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        self.__balance += amount
        self.__add_transaction(f"Deposit", amount)
        return f"Deposited ${amount:.2f}. New balance: ${self.__balance:.2f}"
    
    def withdraw(self, amount):
        """Public method to withdraw money"""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        
        if amount > self.__balance:
            raise ValueError("Insufficient funds")
        
        self.__balance -= amount
        self.__add_transaction(f"Withdrawal", -amount)
        return f"Withdrew ${amount:.2f}. New balance: ${self.__balance:.2f}"
    
    # Property decorator for controlled access to private data
    @property
    def balance(self):
        """Read-only access to balance"""
        return self.__balance
    
    @property
    def transaction_history(self):
        """Read-only access to transaction history"""
        return self.__transaction_history.copy()  # Return a copy, not the original
    
    # Protected method (can be overridden by subclasses)
    def _calculate_interest(self, rate):
        """Protected method for internal calculations"""
        return self.__balance * rate
    
    def __str__(self):
        return f"Account {self.account_number}: Balance ${self.__balance:.2f}"

# Demonstrate encapsulation
print("Creating bank account and demonstrating encapsulation:")
account = BankAccount("ACC123", 1000)

print(f"Account info: {account}")
print(f"Public access to account number: {account.account_number}")
print(f"Property access to balance: ${account.balance:.2f}")

# These operations work through public interface
print("\nUsing public methods:")
print(account.deposit(500))
print(account.withdraw(200))

# Demonstrate access levels
print(f"\nProtected attribute access: {account._account_type}")

# Private attributes are name-mangled
print(f"Private attribute name mangling: {dir(account)}")
print("(Notice __balance becomes _BankAccount__balance)")

# Trying to access private attribute directly (works but shouldn't be done)
print(f"Direct access to private attribute: ${account._BankAccount__balance:.2f}")
print("(This works but violates encapsulation principles)")

# =============================================================================
# 2. INHERITANCE
# =============================================================================

print("\n\n2. INHERITANCE")
print("-" * 40)
print("Inheritance allows a class to inherit attributes and methods from another class,")
print("promoting code reuse and establishing 'is-a' relationships.\n")

# Base class (Parent class)
class Vehicle:
    """Base class representing a generic vehicle"""
    
    def __init__(self, make, model, year, fuel_type):
        self.make = make
        self.model = model
        self.year = year
        self.fuel_type = fuel_type
        self.speed = 0
        self.is_running = False
    
    def start_engine(self):
        """Start the vehicle engine"""
        if not self.is_running:
            self.is_running = True
            return f"{self.year} {self.make} {self.model} engine started"
        return "Engine is already running"
    
    def stop_engine(self):
        """Stop the vehicle engine"""
        if self.is_running and self.speed == 0:
            self.is_running = False
            return f"{self.year} {self.make} {self.model} engine stopped"
        elif self.speed > 0:
            return "Cannot stop engine while moving"
        return "Engine is already off"
    
    def accelerate(self, amount):
        """Accelerate the vehicle"""
        if self.is_running:
            self.speed += amount
            return f"Accelerated to {self.speed} mph"
        return "Cannot accelerate. Engine is off"
    
    def brake(self, amount):
        """Apply brakes"""
        self.speed = max(0, self.speed - amount)
        return f"Braked. Current speed: {self.speed} mph"
    
    def get_info(self):
        """Get vehicle information"""
        return f"{self.year} {self.make} {self.model} ({self.fuel_type})"
    
    def __str__(self):
        return self.get_info()

# Derived class (Child class) - Single Inheritance
class Car(Vehicle):
    """Car class inheriting from Vehicle"""
    
    def __init__(self, make, model, year, fuel_type, doors, transmission):
        # Call parent constructor using super()
        super().__init__(make, model, year, fuel_type)
        
        # Add car-specific attributes
        self.doors = doors
        self.transmission = transmission
        self.air_conditioning = False
    
    # Override parent method (Method Overriding)
    def get_info(self):
        """Override parent method to include car-specific info"""
        base_info = super().get_info()  # Call parent method
        return f"{base_info} - {self.doors} doors, {self.transmission}"
    
    # Add car-specific methods
    def toggle_ac(self):
        """Toggle air conditioning"""
        self.air_conditioning = not self.air_conditioning
        status = "on" if self.air_conditioning else "off"
        return f"Air conditioning turned {status}"
    
    def park(self):
        """Park the car"""
        if self.speed == 0:
            return f"{self.make} {self.model} is parked"
        return "Cannot park while moving. Stop the car first"

# Another derived class
class Motorcycle(Vehicle):
    """Motorcycle class inheriting from Vehicle"""
    
    def __init__(self, make, model, year, fuel_type, engine_size):
        super().__init__(make, model, year, fuel_type)
        self.engine_size = engine_size
        self.has_sidecar = False
    
    def get_info(self):
        """Override to include motorcycle-specific info"""
        base_info = super().get_info()
        sidecar_info = " with sidecar" if self.has_sidecar else ""
        return f"{base_info} - {self.engine_size}cc{sidecar_info}"
    
    def wheelie(self):
        """Perform a wheelie (motorcycle-specific behavior)"""
        if self.is_running and self.speed > 20:
            return f"Performing wheelie on {self.make} {self.model}!"
        return "Need to be running and going faster than 20 mph for wheelie"
    
    def add_sidecar(self):
        """Add a sidecar"""
        self.has_sidecar = True
        return "Sidecar attached"

# Multiple Inheritance Example
class Electric:
    """Mixin class for electric capabilities"""
    
    def __init__(self):
        self.battery_level = 100
        self.charging = False
    
    def charge_battery(self, amount):
        """Charge the battery"""
        if not self.charging:
            self.charging = True
            self.battery_level = min(100, self.battery_level + amount)
            self.charging = False
            return f"Battery charged to {self.battery_level}%"
        return "Already charging"
    
    def get_range(self):
        """Get remaining range based on battery"""
        return f"{self.battery_level * 3} miles remaining"

class ElectricCar(Car, Electric):
    """Electric car using multiple inheritance"""
    
    def __init__(self, make, model, year, doors, transmission):
        Car.__init__(self, make, model, year, "Electric", doors, transmission)
        Electric.__init__(self)
        self.regenerative_braking = True
    
    def brake(self, amount):
        """Override brake to include regenerative braking"""
        result = super().brake(amount)
        if self.regenerative_braking and amount > 0:
            # Regenerative braking adds some charge
            charge_amount = min(5, amount)
            self.battery_level = min(100, self.battery_level + charge_amount)
            result += f" (Regenerative braking: +{charge_amount}% battery)"
        return result
    
    def get_info(self):
        """Override to include electric car info"""
        car_info = Car.get_info(self)
        return f"{car_info} - Battery: {self.battery_level}%"

# Demonstrate inheritance
print("Creating vehicles to demonstrate inheritance:")

# Base class instance
generic_vehicle = Vehicle("Generic", "Vehicle", 2020, "Gasoline")
print(f"Generic vehicle: {generic_vehicle}")

# Single inheritance
car = Car("Toyota", "Camry", 2023, "Hybrid", 4, "Automatic")
print(f"Car: {car}")
print(car.start_engine())
print(car.accelerate(30))
print(car.toggle_ac())

# Another single inheritance
motorcycle = Motorcycle("Harley-Davidson", "Sportster", 2022, "Gasoline", 1200)
print(f"\nMotorcycle: {motorcycle}")
print(motorcycle.start_engine())
print(motorcycle.accelerate(25))
print(motorcycle.wheelie())

# Multiple inheritance
electric_car = ElectricCar("Tesla", "Model 3", 2023, 4, "Automatic")
print(f"\nElectric car: {electric_car}")
print(electric_car.start_engine())
print(electric_car.accelerate(40))
print(electric_car.brake(15))
print(electric_car.charge_battery(10))
print(electric_car.get_range())

# Method Resolution Order (MRO)
print(f"\nElectric car MRO: {ElectricCar.__mro__}")

# =============================================================================
# 3. POLYMORPHISM
# =============================================================================

print("\n\n3. POLYMORPHISM")
print("-" * 40)
print("Polymorphism allows objects of different types to be treated uniformly")
print("through a common interface, with each type providing its own implementation.\n")

# Base class for polymorphism demonstration
class Animal:
    """Base class for animals"""
    
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        """Method to be overridden by subclasses"""
        return f"{self.name} makes a generic animal sound"
    
    def move(self):
        """Method to be overridden by subclasses"""
        return f"{self.name} moves around"
    
    def eat(self):
        """Common method for all animals"""
        return f"{self.name} is eating"
    
    def __str__(self):
        return f"{self.name} the {self.species}"

# Polymorphic subclasses
class Dog(Animal):
    """Dog class demonstrating polymorphism"""
    
    def __init__(self, name, breed):
        super().__init__(name, "Dog")
        self.breed = breed
    
    def make_sound(self):
        """Dog-specific implementation"""
        return f"{self.name} barks: Woof! Woof!"
    
    def move(self):
        """Dog-specific implementation"""
        return f"{self.name} runs around on four legs"
    
    def fetch(self):
        """Dog-specific method"""
        return f"{self.name} fetches the ball"

class Cat(Animal):
    """Cat class demonstrating polymorphism"""
    
    def __init__(self, name, color):
        super().__init__(name, "Cat")
        self.color = color
    
    def make_sound(self):
        """Cat-specific implementation"""
        return f"{self.name} meows: Meow! Meow!"
    
    def move(self):
        """Cat-specific implementation"""
        return f"{self.name} gracefully prowls around"
    
    def purr(self):
        """Cat-specific method"""
        return f"{self.name} purrs contentedly"

class Bird(Animal):
    """Bird class demonstrating polymorphism"""
    
    def __init__(self, name, wing_span):
        super().__init__(name, "Bird")
        self.wing_span = wing_span
    
    def make_sound(self):
        """Bird-specific implementation"""
        return f"{self.name} chirps: Tweet! Tweet!"
    
    def move(self):
        """Bird-specific implementation"""
        return f"{self.name} flies through the air"
    
    def fly(self):
        """Bird-specific method"""
        return f"{self.name} soars with {self.wing_span} inch wingspan"

class Fish(Animal):
    """Fish class demonstrating polymorphism"""
    
    def __init__(self, name, water_type):
        super().__init__(name, "Fish")
        self.water_type = water_type
    
    def make_sound(self):
        """Fish-specific implementation"""
        return f"{self.name} makes bubble sounds: Blub! Blub!"
    
    def move(self):
        """Fish-specific implementation"""
        return f"{self.name} swims through {self.water_type} water"
    
    def swim(self):
        """Fish-specific method"""
        return f"{self.name} glides through the water"

# Function demonstrating polymorphism
def animal_interactions(animals):
    """
    Function that works with any animal object (polymorphism)
    Each animal type will behave differently despite using the same interface
    """
    print("Animal interactions demonstration:")
    print("-" * 30)
    
    for animal in animals:
        print(f"\n{animal}")
        print(f"  Sound: {animal.make_sound()}")
        print(f"  Movement: {animal.move()}")
        print(f"  Eating: {animal.eat()}")
        
        # Type-specific behavior (checking type for additional methods)
        if isinstance(animal, Dog):
            print(f"  Special: {animal.fetch()}")
        elif isinstance(animal, Cat):
            print(f"  Special: {animal.purr()}")
        elif isinstance(animal, Bird):
            print(f"  Special: {animal.fly()}")
        elif isinstance(animal, Fish):
            print(f"  Special: {animal.swim()}")

# Operator overloading (another form of polymorphism)
class Vector:
    """Vector class demonstrating operator overloading"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Overload + operator"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Overload - operator"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Overload * operator for scalar multiplication"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        """Overload == operator"""
        return self.x == other.x and self.y == other.y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        return self.__str__()

# Demonstrate polymorphism
print("Creating animals for polymorphism demonstration:")

animals = [
    Dog("Buddy", "Golden Retriever"),
    Cat("Whiskers", "Orange"),
    Bird("Tweety", 12),
    Fish("Nemo", "salt")
]

# Polymorphic function call - same function works with different types
animal_interactions(animals)

# Operator overloading demonstration
print("\n\nOperator Overloading (Polymorphism with operators):")
print("-" * 50)

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(f"Vector 1: {v1}")
print(f"Vector 2: {v2}")
print(f"Addition: {v1} + {v2} = {v1 + v2}")
print(f"Subtraction: {v1} - {v2} = {v1 - v2}")
print(f"Scalar multiplication: {v1} * 2 = {v1 * 2}")
print(f"Equality: {v1} == {v2} is {v1 == v2}")

# =============================================================================
# 4. ABSTRACTION
# =============================================================================

print("\n\n4. ABSTRACTION")
print("-" * 40)
print("Abstraction hides implementation complexity and provides a simplified interface.")
print("Python uses abstract base classes (ABC) to define interfaces.\n")

from abc import ABC, abstractmethod

# Abstract base class
class Shape(ABC):
    """Abstract base class for shapes"""
    
    def __init__(self, color="black"):
        self.color = color
    
    @abstractmethod
    def area(self):
        """Abstract method - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Abstract method - must be implemented by subclasses"""
        pass
    
    # Concrete method (shared implementation)
    def get_info(self):
        """Concrete method available to all subclasses"""
        return f"{self.__class__.__name__} (color: {self.color})"
    
    def paint(self, new_color):
        """Concrete method to change color"""
        old_color = self.color
        self.color = new_color
        return f"Changed color from {old_color} to {new_color}"

# Concrete implementations of abstract class
class Rectangle(Shape):
    """Rectangle implementation of Shape abstract class"""
    
    def __init__(self, width, height, color="black"):
        super().__init__(color)
        self.width = width
        self.height = height
    
    def area(self):
        """Implementation of abstract area method"""
        return self.width * self.height
    
    def perimeter(self):
        """Implementation of abstract perimeter method"""
        return 2 * (self.width + self.height)
    
    def __str__(self):
        return f"Rectangle({self.width}x{self.height}, {self.color})"

class Circle(Shape):
    """Circle implementation of Shape abstract class"""
    
    def __init__(self, radius, color="black"):
        super().__init__(color)
        self.radius = radius
    
    def area(self):
        """Implementation of abstract area method"""
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        """Implementation of abstract perimeter method"""
        import math
        return 2 * math.pi * self.radius
    
    def __str__(self):
        return f"Circle(radius={self.radius}, {self.color})"

class Triangle(Shape):
    """Triangle implementation of Shape abstract class"""
    
    def __init__(self, side1, side2, side3, color="black"):
        super().__init__(color)
        self.side1 = side1
        self.side2 = side2
        self.side3 = side3
    
    def area(self):
        """Implementation using Heron's formula"""
        s = self.perimeter() / 2  # semi-perimeter
        import math
        return math.sqrt(s * (s - self.side1) * (s - self.side2) * (s - self.side3))
    
    def perimeter(self):
        """Implementation of abstract perimeter method"""
        return self.side1 + self.side2 + self.side3
    
    def __str__(self):
        return f"Triangle({self.side1}, {self.side2}, {self.side3}, {self.color})"

# Abstract class for data processing (another abstraction example)
class DataProcessor(ABC):
    """Abstract class for data processing"""
    
    def __init__(self, data):
        self.data = data
        self.processed_data = None
    
    def process(self):
        """Template method - defines the algorithm structure"""
        print(f"Starting {self.__class__.__name__} processing...")
        self.validate_data()
        self.processed_data = self.transform_data()
        self.save_results()
        print("Processing complete!")
        return self.processed_data
    
    def validate_data(self):
        """Concrete method - common validation"""
        if not self.data:
            raise ValueError("No data to process")
        print(f"Data validation passed: {len(self.data)} items")
    
    @abstractmethod
    def transform_data(self):
        """Abstract method - specific transformation logic"""
        pass
    
    def save_results(self):
        """Concrete method - common save operation"""
        print(f"Results saved: {len(self.processed_data)} processed items")

# Concrete data processors
class TextProcessor(DataProcessor):
    """Text processing implementation"""
    
    def transform_data(self):
        """Implement text-specific transformation"""
        print("Applying text transformations...")
        return [text.upper().strip() for text in self.data if isinstance(text, str)]

class NumberProcessor(DataProcessor):
    """Number processing implementation"""
    
    def transform_data(self):
        """Implement number-specific transformation"""
        print("Applying mathematical transformations...")
        return [num * 2 for num in self.data if isinstance(num, (int, float))]

# Function that works with any shape (abstraction in action)
def calculate_shape_properties(shapes):
    """Function that works with any shape through abstraction"""
    print("Shape calculations using abstraction:")
    print("-" * 40)
    
    total_area = 0
    total_perimeter = 0
    
    for shape in shapes:
        area = shape.area()
        perimeter = shape.perimeter()
        total_area += area
        total_perimeter += perimeter
        
        print(f"{shape}")
        print(f"  Area: {area:.2f}")
        print(f"  Perimeter: {perimeter:.2f}")
        print(f"  Info: {shape.get_info()}")
        print()
    
    print(f"Total area of all shapes: {total_area:.2f}")
    print(f"Total perimeter of all shapes: {total_perimeter:.2f}")

# Demonstrate abstraction
print("Creating shapes to demonstrate abstraction:")

# Cannot instantiate abstract class
try:
    abstract_shape = Shape()  # This will raise TypeError
except TypeError as e:
    print(f"Cannot instantiate abstract class: {e}")

# Create concrete implementations
shapes = [
    Rectangle(5, 3, "red"),
    Circle(4, "blue"),
    Triangle(3, 4, 5, "green")
]

# Use shapes through abstract interface
calculate_shape_properties(shapes)

# Demonstrate painting (concrete method from abstract class)
print("Demonstrating concrete methods from abstract class:")
for shape in shapes:
    print(shape.paint("yellow"))

# Data processing abstraction
print("\n\nData Processing Abstraction:")
print("-" * 30)

text_data = ["  hello  ", "  WORLD  ", "  python  ", ""]
number_data = [1, 2.5, 3, 4.7, 0]

text_processor = TextProcessor(text_data)
number_processor = NumberProcessor(number_data)

# Both use the same interface but different implementations
text_result = text_processor.process()
print(f"Text processing result: {text_result}")

print()
number_result = number_processor.process()
print(f"Number processing result: {number_result}")

# =============================================================================
# COMPREHENSIVE EXAMPLE: PUTTING IT ALL TOGETHER
# =============================================================================

print("\n\n" + "="*70)
print("COMPREHENSIVE EXAMPLE: ALL OOP CONCEPTS TOGETHER")
print("="*70)

class Employee(ABC):
    """Abstract base class for employees (Abstraction)"""
    
    # Class variable (shared by all instances)
    company_name = "TechCorp Inc."
    employee_count = 0
    
    def __init__(self, employee_id, name, base_salary):
        # Encapsulation: private and protected attributes
        self.__employee_id = employee_id  # Private
        self._name = name                 # Protected
        self.__base_salary = base_salary  # Private
        self._department = None           # Protected
        
        Employee.employee_count += 1
    
    # Abstract method (must be implemented by subclasses)
    @abstractmethod
    def calculate_salary(self):
        """Calculate total salary including bonuses"""
        pass
    
    @abstractmethod
    def get_role_description(self):
        """Get description of the role"""
        pass
    
    # Property decorators for controlled access (Encapsulation)
    @property
    def employee_id(self):
        return self.__employee_id
    
    @property
    def name(self):
        return self._name
    
    @property
    def base_salary(self):
        return self.__base_salary
    
    @base_salary.setter
    def base_salary(self, new_salary):
        if new_salary < 0:
            raise ValueError("Salary cannot be negative")
        self.__base_salary = new_salary
    
    # Common methods for all employees
    def get_employee_info(self):
        """Get basic employee information"""
        return f"ID: {self.__employee_id}, Name: {self._name}, Department: {self._department}"
    
    def assign_department(self, department):
        """Assign employee to a department"""
        self._department = department
        return f"{self._name} assigned to {department} department"
    
    def __str__(self):
        return f"{self._name} ({self.__class__.__name__})"

# Inheritance: Developer inherits from Employee
class Developer(Employee):
    """Developer class inheriting from Employee"""
    
    def __init__(self, employee_id, name, base_salary, programming_languages):
        super().__init__(employee_id, name, base_salary)
        self.programming_languages = programming_languages
        self._department = "Engineering"
        self.__project_bonus = 0
    
    # Implementation of abstract method (Abstraction)
    def calculate_salary(self):
        """Calculate developer salary with project bonus"""
        return self.base_salary + self.__project_bonus
    
    def get_role_description(self):
        """Implementation of abstract method"""
        langs = ", ".join(self.programming_languages)
        return f"Software Developer specializing in: {langs}"
    
    # Developer-specific methods
    def add_project_bonus(self, amount):
        """Add project completion bonus"""
        self.__project_bonus += amount
        return f"Added ${amount} project bonus. Total bonus: ${self.__project_bonus}"
    
    def learn_technology(self, technology):
        """Learn new programming language/technology"""
        if technology not in self.programming_languages:
            self.programming_languages.append(technology)
            return f"{self.name} learned {technology}"
        return f"{self.name} already knows {technology}"

class Manager(Employee):
    """Manager class inheriting from Employee"""
    
    def __init__(self, employee_id, name, base_salary, team_size):
        super().__init__(employee_id, name, base_salary)
        self.team_size = team_size
        self._department = "Management"
        self.__team_members = []
        self.__management_bonus = 0
    
    def calculate_salary(self):
        """Calculate manager salary with team bonus"""
        team_bonus = self.team_size * 1000  # $1000 per team member
        return self.base_salary + team_bonus + self.__management_bonus
    
    def get_role_description(self):
        """Implementation of abstract method"""
        return f"Manager leading a team of {self.team_size} people"
    
    # Manager-specific methods
    def add_team_member(self, employee):
        """Add team member"""
        self.__team_members.append(employee)
        self.team_size = len(self.__team_members)
        return f"Added {employee.name} to {self.name}'s team"
    
    def give_performance_bonus(self, amount):
        """Give performance bonus"""
        self.__management_bonus += amount
        return f"Performance bonus of ${amount} added"

class SalesRep(Employee):
    """Sales Representative class"""
    
    def __init__(self, employee_id, name, base_salary, commission_rate):
        super().__init__(employee_id, name, base_salary)
        self.commission_rate = commission_rate  # percentage
        self._department = "Sales"
        self.__total_sales = 0
    
    def calculate_salary(self):
        """Calculate salary with commission"""
        commission = self.__total_sales * (self.commission_rate / 100)
        return self.base_salary + commission
    
    def get_role_description(self):
        """Implementation of abstract method"""
        return f"Sales Representative with {self.commission_rate}% commission rate"
    
    def record_sale(self, amount):
        """Record a sale"""
        self.__total_sales += amount
        commission = amount * (self.commission_rate / 100)
        return f"Sale recorded: ${amount}, Commission earned: ${commission:.2f}"
    
    @property
    def total_sales(self):
        return self.__total_sales

# Polymorphism demonstration function
def process_payroll(employees):
    """Process payroll for all employees (Polymorphism)"""
    print("Processing payroll for all employees:")
    print("-" * 50)
    
    total_payroll = 0
    
    for employee in employees:
        # Polymorphism: same method call, different implementations
        salary = employee.calculate_salary()
        role = employee.get_role_description()
        
        total_payroll += salary
        
        print(f"{employee}")
        print(f"  Role: {role}")
        print(f"  Salary: ${salary:,.2f}")
        print(f"  Info: {employee.get_employee_info()}")
        print()
    
    print(f"Total payroll: ${total_payroll:,.2f}")
    print(f"Average salary: ${total_payroll / len(employees):,.2f}")
    return total_payroll

# Demonstration of all concepts together
print("Creating employees to demonstrate all OOP concepts:")

# Create different types of employees
employees = [
    Developer("DEV001", "Alice Chen", 80000, ["Python", "JavaScript", "React"]),
    Developer("DEV002", "Bob Wilson", 85000, ["Java", "Spring", "MySQL"]),
    Manager("MGR001", "Carol Smith", 95000, 5),
    SalesRep("SAL001", "David Brown", 50000, 5.5),
    SalesRep("SAL002", "Eva Martinez", 52000, 6.0)
]

# Demonstrate specific behaviors
print("Demonstrating specific employee behaviors:")
alice = employees[0]  # Developer
carol = employees[2]  # Manager
david = employees[3]  # Sales rep

print(alice.learn_technology("Docker"))
print(alice.add_project_bonus(5000))

print(carol.add_team_member(alice))
print(carol.give_performance_bonus(3000))

print(david.record_sale(25000))
print(david.record_sale(15000))

# Polymorphism in action
print(f"\nPolymorphism demonstration:")
total_payroll = process_payroll(employees)

# Class information
print(f"\nCompany: {Employee.company_name}")
print(f"Total employees: {Employee.employee_count}")

print(f"\nOOP Concepts Demonstrated:")
print("1. Encapsulation: Private attributes, property decorators, controlled access")
print("2. Inheritance: Developer, Manager, SalesRep inherit from Employee")
print("3. Polymorphism: Same interface (calculate_salary), different implementations")
print("4. Abstraction: Abstract Employee class with concrete implementations")