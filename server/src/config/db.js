/**
 * db.js — MongoDB Connection Setup
 *
 * Connects to MongoDB using Mongoose.
 * The connection URI comes from the .env file.
 */

const mongoose = require("mongoose");

const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGO_URI);
    console.log(`✅ MongoDB connected: ${conn.connection.host}`);
  } catch (error) {
    console.error(`❌ MongoDB connection error: ${error.message}`);
    process.exit(1); // Exit the process if DB connection fails
  }
};

module.exports = connectDB;
