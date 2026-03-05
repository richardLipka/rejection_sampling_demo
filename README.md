# Setup and Deployment

## Prerequisites

Ensure you have the following installed on your system:

- **Node.js** (version 18 or higher recommended)
- **npm** (comes with Node.js) or **yarn**

## Prepare the Project

If you have the source code locally:

1. Open a terminal or command prompt.
2. Navigate to the project's root directory (the directory containing `package.json`).

## Install Dependencies

Install all required libraries (such as `recharts`, `lucide-react`, `framer-motion`, etc.):

```bash
npm install```

## Running in Development Mode

To run the application locally for development or testing:

```bash
npm run dev

By default, Vite will start the development server at:

http://localhost:5173

(or port 3000 if configured in vite.config.ts).

## Building for Production

To create an optimized production build:

```bash
npm run build

This command generates a dist/ folder containing optimized HTML, CSS, and JavaScript files that can be deployed to a web server.
