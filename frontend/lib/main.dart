import 'package:flutter/material.dart';

void main() {
  runApp(FitBalanceApp());
}

class FitBalanceApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FitBalance',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: FitBalanceHomePage(),
    );
  }
}

class FitBalanceHomePage extends StatefulWidget {
  @override
  _FitBalanceHomePageState createState() => _FitBalanceHomePageState();
}

class _FitBalanceHomePageState extends State<FitBalanceHomePage> {
  int _selectedIndex = 0;

  final List<Widget> _pages = [
    BiomechanicsPage(),
    NutritionPage(),
    BurnoutPage(),
    ProfilePage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('FitBalance AI'),
        backgroundColor: Colors.blue[600],
      ),
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        currentIndex: _selectedIndex,
        onTap: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        items: [
          BottomNavigationBarItem(
            icon: Icon(Icons.fitness_center),
            label: 'Biomechanics',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.restaurant),
            label: 'Nutrition',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.psychology),
            label: 'Burnout',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person),
            label: 'Profile',
          ),
        ],
      ),
    );
  }
}

class BiomechanicsPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.fitness_center, size: 100, color: Colors.blue),
          SizedBox(height: 20),
          Text(
            'Biomechanics Analysis',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 10),
          Text('Real-time movement analysis with AI'),
          SizedBox(height: 30),
          ElevatedButton(
            onPressed: () {
              // TODO: Implement video recording and analysis
            },
            child: Text('Record Exercise'),
          ),
        ],
      ),
    );
  }
}

class NutritionPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.restaurant, size: 100, color: Colors.green),
          SizedBox(height: 20),
          Text(
            'Nutrition Analysis',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 10),
          Text('AI-powered meal analysis and recommendations'),
          SizedBox(height: 30),
          ElevatedButton(
            onPressed: () {
              // TODO: Implement photo capture and analysis
            },
            child: Text('Analyze Meal'),
          ),
        ],
      ),
    );
  }
}

class BurnoutPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.psychology, size: 100, color: Colors.orange),
          SizedBox(height: 20),
          Text(
            'Burnout Prediction',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 10),
          Text('AI-powered burnout risk assessment'),
          SizedBox(height: 30),
          ElevatedButton(
            onPressed: () {
              // TODO: Implement burnout assessment
            },
            child: Text('Assess Risk'),
          ),
        ],
      ),
    );
  }
}

class ProfilePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.person, size: 100, color: Colors.purple),
          SizedBox(height: 20),
          Text(
            'User Profile',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 10),
          Text('Personalized fitness insights and progress'),
          SizedBox(height: 30),
          ElevatedButton(
            onPressed: () {
              // TODO: Implement profile management
            },
            child: Text('View Profile'),
          ),
        ],
      ),
    );
  }
} 