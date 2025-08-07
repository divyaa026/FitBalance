const functions = require('firebase-functions');
const admin = require('firebase-admin');

// Initialize Firebase services
admin.initializeApp();

// Cloud Functions for FitBalance
exports.fitbalance = functions.https.onRequest((req, res) => {
  res.json({
    message: 'FitBalance Firebase Functions',
    status: 'active',
    timestamp: new Date().toISOString()
  });
});

// Cloud Firestore + Auth triggers
exports.onUserCreate = functions.auth.user().onCreate((user) => {
  return admin.firestore().collection("users").doc(user.uid).set({
    createdAt: new Date(),
    burnoutRisk: "low" // Initial state
  });
});

// User authentication function
exports.authenticateUser = functions.https.onCall(async (data, context) => {
  try {
    const { uid, email } = data;
    
    // Create or update user profile
    await admin.firestore().collection('users').doc(uid).set({
      email: email,
      createdAt: admin.firestore.FieldValue.serverTimestamp(),
      lastLogin: admin.firestore.FieldValue.serverTimestamp()
    }, { merge: true });
    
    return {
      success: true,
      message: 'User authenticated successfully'
    };
  } catch (error) {
    throw new functions.https.HttpsError('internal', error.message);
  }
});

// Store biomechanics analysis
exports.storeBiomechanicsAnalysis = functions.https.onCall(async (data, context) => {
  try {
    const { userId, analysisData } = data;
    
    await admin.firestore().collection('biomechanics_analyses').add({
      userId: userId,
      analysis: analysisData,
      timestamp: admin.firestore.FieldValue.serverTimestamp()
    });
    
    return {
      success: true,
      message: 'Biomechanics analysis stored'
    };
  } catch (error) {
    throw new functions.https.HttpsError('internal', error.message);
  }
});

// Store nutrition analysis
exports.storeNutritionAnalysis = functions.https.onCall(async (data, context) => {
  try {
    const { userId, mealData } = data;
    
    await admin.firestore().collection('nutrition_analyses').add({
      userId: userId,
      meal: mealData,
      timestamp: admin.firestore.FieldValue.serverTimestamp()
    });
    
    return {
      success: true,
      message: 'Nutrition analysis stored'
    };
  } catch (error) {
    throw new functions.https.HttpsError('internal', error.message);
  }
});

// Store burnout analysis
exports.storeBurnoutAnalysis = functions.https.onCall(async (data, context) => {
  try {
    const { userId, riskData } = data;
    
    await admin.firestore().collection('burnout_analyses').add({
      userId: userId,
      risk: riskData,
      timestamp: admin.firestore.FieldValue.serverTimestamp()
    });
    
    return {
      success: true,
      message: 'Burnout analysis stored'
    };
  } catch (error) {
    throw new functions.https.HttpsError('internal', error.message);
  }
});

// Scheduled function for data cleanup
exports.cleanupOldData = functions.pubsub.schedule('every 24 hours').onRun(async (context) => {
  try {
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
    
    // Clean up old analyses
    const collections = ['biomechanics_analyses', 'nutrition_analyses', 'burnout_analyses'];
    
    for (const collectionName of collections) {
      const snapshot = await admin.firestore()
        .collection(collectionName)
        .where('timestamp', '<', thirtyDaysAgo)
        .get();
      
      const batch = admin.firestore().batch();
      snapshot.docs.forEach((doc) => {
        batch.delete(doc.ref);
      });
      
      await batch.commit();
    }
    
    console.log('Data cleanup completed');
    return null;
  } catch (error) {
    console.error('Data cleanup error:', error);
    return null;
  }
}); 