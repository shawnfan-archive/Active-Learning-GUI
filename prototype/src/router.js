import Vue from 'vue'
import VueRouter from 'vue-router'
import ActiveLearning from './components/ActiveLearning'

Vue.use(VueRouter)

export default new VueRouter({
    routes: [{
        path: '/', 
        component: ActiveLearning
    }],
    mode: 'history'
})