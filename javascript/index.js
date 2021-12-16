// smooth scrolling effect
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();

        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// https://www.chartjs.org/docs/latest/charts/doughnut.html#pie
const data = {
    labels: [
        'Image MCQ',
        'Text MCQ',
        'Fill-in-the-blank'
    ],
    datasets: [{
        label: 'Task Distr.',
        data: [57672, 31578, 18189],
        backgroundColor: [
        'rgb(255, 99, 132)',
        'rgb(54, 162, 235)',
        'rgb(255, 205, 86)'
        ],
        // pop when hovered
        // https://www.chartjs.org/docs/latest/charts/doughnut.html#interactions
        hoverOffset: 10 
    }]
};
const config = {
    type: 'pie',
    data: data,
    // makes sure the chart doesn't get clipped when a section is hovered.
    options: {
        plugins: {
            legend: {
                display: true,
                labels: {
                    font: {
                        size: 15
                    }
                }
            },
        },
        layout: {
            padding: {
               left: 5,
               right: 5,
               bottom: 5
            }
        },
    }
};
//  painting the chart onto the webpage.
var myChart = new Chart(
    document.getElementById('task-pie-chart'),
    config
);