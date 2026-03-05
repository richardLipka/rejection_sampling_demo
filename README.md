1. Prerequisites
Ensure you have the following installed on your server:
Node.js (Version 18 or higher recommended)
npm (comes with Node.js) or yarn

2. Prepare the Project
If you have the source code files locally:
Open your terminal or command prompt.
Navigate to the project's root directory (where package.json is located).

3. Install Dependencies
Run the following command to install all necessary libraries (recharts, lucide-react, framer-motion, etc.):

code
Bash
npm install

4. Running in Development Mode
If you want to run the app for testing or development purposes:

code
Bash
npm run dev

By default, Vite will start the server at http://localhost:5173 (or port 3000 if configured in vite.config.ts).

6. Building for Production
To deploy the app to a live server, you should create an optimized production build:

code
Bash
npm run build

This command creates a dist/ folder containing highly optimized HTML, CSS, and JavaScript files.
