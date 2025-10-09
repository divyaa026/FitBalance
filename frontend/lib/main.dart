import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'pages/nutrition.dart';
import 'pages/biomechanics.dart';
import 'pages/burnout.dart';
import 'providers/nutrition_provider.dart';
import 'providers/biomechanics_provider.dart';
import 'providers/burnout_provider.dart';

void main() {
  runApp(FitBalanceApp());
}

class FitBalanceApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => NutritionProvider()),
        ChangeNotifierProvider(create: (_) => BiomechanicsProvider()),
        ChangeNotifierProvider(create: (_) => BurnoutProvider()),
      ],
      child: MaterialApp(
        title: 'FitBalance',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          visualDensity: VisualDensity.adaptivePlatformDensity,
        ),
        home: FitBalanceHomePage(),
      ),
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
    BiomechanicsPage(), // Now using the proper BiomechanicsPage with backend integration
    NutritionPage(), // Now using the proper NutritionPage with backend integration
    BurnoutPage(), // Now using the proper BurnoutPage with backend integration
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