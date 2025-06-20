# 🏀 NBA Draft Big Board

**A comprehensive Streamlit application for evaluating and ranking NBA draft prospects**

![Version](https://img.shields.io/badge/version-1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation System](#evaluation-system)
- [Data Requirements](#data-requirements)
- [File Structure](#file-structure)
- [Contributing](#contributing)

---

## 🎯 Overview

The NBA Draft Big Board is a sophisticated scouting application that allows users to evaluate, rank, and compare NBA draft prospects using a comprehensive 9-category evaluation system. Built with Streamlit, it provides an intuitive interface for scouts, analysts, and basketball enthusiasts to create their own draft boards.

### ✨ Key Highlights

- **Multi-dimensional Evaluation**: 9 comprehensive categories covering all aspects of basketball performance
- **Interactive Visualizations**: Radar charts and statistical comparisons
- **Real-time Rankings**: Automatic sorting based on weighted scores
- **Data Integration**: Seamless integration with basketball statistical data from [Tankathon](https://tankathon.com/big-board)
    
- **Export Functionality**: Save rankings to text files for sharing

---

## 🚀 Features

### 🎛️ Core Functionality

| Feature | Description |
|---------|-------------|
| **Player Evaluation** | Rate prospects across 9 key basketball categories |
| **Weighted Scoring** | Intelligent scoring system with balanced category weights |
| **Tier Classification** | Automatic tier assignment (Superstar to Role Player) |
| **Interactive Rankings** | Live-updating big board with drag-and-drop functionality |

### 📊 Analytics & Visualization

- **Radar Charts**: Multi-player overlay comparisons
- **Statistical Analysis**: Advanced metrics integration
- **Performance Highlighting**: Visual emphasis on standout statistics
- **Export Options**: Multiple format support for data sharing

### 💾 Data Management

- **Auto-save**: Automatic progress preservation
- **Import/Export**: JSON and text file support
- **CSV Integration**: Statistics data loading
- **Session Persistence**: Maintains state across sessions

---

## 🛠️ Installation

### Prerequisites

```bash
Python 3.8+
pip package manager
```

### Required Dependencies

```bash
pip install streamlit pandas matplotlib numpy
```

### Quick Start

1. **Clone or download** the application files
2. **Prepare your data** - Place `nba_prospects_2025_stats.csv` in the root directory
3. **Run the application**:
   ```bash
   streamlit run app.py
   ```
4. **Access the app** at `http://localhost:8501`

---

## 🎮 Usage

### Getting Started

1. **Launch the application** using the Streamlit command
2. **Select a player** from the dropdown (populated from your CSV data)
3. **Evaluate the player** using the 9-category slider system
4. **Add to Big Board** to see real-time ranking updates

### Step-by-Step Evaluation Process

#### 1. Player Selection
- Choose from the dropdown of available prospects
- View auto-populated player information (age, height, position, team)

#### 2. Category Evaluation
Rate each player on a scale of 0-10 across these categories:

| Category | Weight | Description |
|----------|--------|-------------|
| **Athleticism** | 11.1% | Speed, jumping ability, coordination |
| **Scoring** | 11.1% | Ability to put the ball in the basket |
| **Shooting** | 11.1% | Range, accuracy, shooting mechanics |
| **Dribbling** | 11.1% | Ball handling skills and control |
| **Passing** | 11.1% | Vision, accuracy, decision-making |
| **Perimeter Defense** | 11.1% | On-ball defense, lateral quickness |
| **Interior Defense** | 11.1% | Shot blocking, post defense, rebounding |
| **Basketball IQ** | 11.1% | Game understanding, situational awareness |
| **Intangibles** | 11.1% | Leadership, work ethic, character |

#### 3. Review and Submit
- Preview your weighted score and tier assignment
- Add the player to your big board
- View updated rankings in real-time

---

## 📈 Evaluation System

### Scoring Methodology

The application uses a **weighted average system** where each category contributes equally (11.1%) to the final score:

```python
Final Score = Σ(Category Score × Weight)
```

### Tier Classification

| Tier | Score Range | Description |
|------|-------------|-------------|
| **Tier 1** | 9.0 - 10.0 | Superstar Potential |
| **Tier 2** | 8.0 - 8.9 | Potential All-NBA |
| **Tier 3** | 7.0 - 7.9 | Potential All-Star |
| **Tier 4** | 6.0 - 6.9 | Quality Starter |
| **Tier 5** | 0.0 - 5.9 | Role Player |

### Advanced Features

- **Custom Tier Override**: Manual tier adjustments for subjective factors
- **Statistical Integration**: NCAA performance metrics influence
- **Comparison Tools**: Side-by-side player analysis

---

## 📁 Data Requirements

### Primary Data File

**File**: `nba_prospects_2025_stats.csv`

**Required Columns**:
```
name, team, year, position, measurements, age_at_draft, nation, 
wingspan, games, minutes_per_game, 3p_pct, ft_pct, rebounds_per36, 
assists_per36, blocks_per36, steals_per36, turnovers_per36, 
points_per36, ts_per, 3pa_rate, fta_rate, usg_per, ast_per_usg, 
ast_per_to, obpm, dbpm, bpm
```

### Data Sources

- **NCAA Statistics**: Official college basketball metrics
- **Scouting Reports**: Professional evaluation data
- **Combine Measurements**: Physical and athletic testing results

---

## 🏗️ File Structure

```
nba-draft-app/
├── app.py                          # Main application file
├── nba_prospects_2025_stats.csv    # Player statistics data
├── big_board_save.json            # Auto-saved rankings
├── big_board_nba_draft_2025.txt   # Exported rankings
└── README.md                      # This documentation
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `load_data()` | CSV data loading and preprocessing |
| `calculate_weighted_average()` | Score computation |
| `create_overlaid_radar_chart()` | Visualization generation |
| `get_player_stats()` | Statistical data retrieval |
| `save_big_board_to_txt()` | Export functionality |

---

## 🎨 User Interface

### Main Dashboard
- **Header**: Application title and branding
- **Sidebar**: Controls for save/load and weight display
- **Main Panel**: Player evaluation form and rankings display

### Interactive Elements
- **Sliders**: 0-10 evaluation scales for each category
- **Data Editor**: Editable rankings table with tier override
- **Comparison Tools**: Multi-player analysis interface

### Visual Design
- **Modern Aesthetics**: Clean, professional interface
- **Color Coding**: Performance highlighting (green/red)
- **Responsive Layout**: Adapts to different screen sizes

---

## 🔧 Customization

### Modifying Evaluation Categories

To add or modify categories, update the `EVAL_CATEGORIES` and `WEIGHTS` constants:

```python
EVAL_CATEGORIES = [
    "Athleticism", "Dribbling", "Shooting", 
    # Add your custom categories here
]

WEIGHTS = {
    "Athleticism": 1/9,  # Adjust weights as needed
    # Add corresponding weights
}
```

### Styling Customization

- **Colors**: Modify highlighting colors in the `apply_highlighting()` function
- **Layout**: Adjust column ratios and spacing in the Streamlit layout
- **Charts**: Customize radar chart appearance in `create_overlaid_radar_chart()`

---

## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **CSV not loading** | Verify file name and column headers |
| **Player not found** | Check name spelling and CSV data |
| **Radar chart errors** | Ensure matplotlib backend compatibility |
| **Save/load issues** | Check file permissions and disk space |

### Performance Tips

- **Large datasets**: Consider data filtering for better performance
- **Memory usage**: Clear big board periodically to free memory
- **Visualization**: Limit radar chart comparisons to 4 players maximum

---

## 🤝 Contributing

We welcome contributions to improve the NBA Draft Big Board application!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your improvements**
4. **Test thoroughly**
5. **Submit a pull request**

### Areas for Enhancement

- [ ] Advanced statistical models
- [ ] Machine learning integration
- [ ] Mobile responsiveness improvements
- [ ] Additional export formats
- [ ] User authentication system

---

## 📞 Support

### Getting Help

- **Documentation**: Refer to this README for detailed guidance
- **Issues**: Report bugs and feature requests
- **Community**: Join discussions about basketball analytics

### Contact Information

For technical support or feature requests, please create an issue in the project repository.

---

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Scouting! 🏀**

*Build your ultimate NBA Draft Big Board and discover the next generation of basketball talent.*